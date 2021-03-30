from data_loader import load_npy, load_data_filename
from models import auto_encoder
from summary_ops import scalar_summary, images_summary
from inpaint_ops import random_bbox, bbox2mask
from tqdm import tqdm

import tensorflow as tf
import os.path as op
import datetime
import math

print(tf.__version__)


class hp:
    # Training setting
    training_file = './data/train_list.txt'
    testing_file = './data/test_list.txt'
    logdir = './logs'
    checkpoint_dir = './checkpoints/ckpt_2rd'
    checkpoint_prefix = op.join(checkpoint_dir, "ckpt")
    checkpoint_restore_dir = './checkpoints/ckpt'
    checkpoint_freq = 2
    epochs = 50
    steps_per_epoch = -1 # -1: whole training data
    validation_split = 0.1
    batch = True
    batch_size = 32
    max_outputs = 10
    profile = True # profile on first epoch, batch 10~20
    # Data
    image_height = 256
    image_width = 256
    mask_height = 256
    mask_width = 96
    max_delta_height = 0
    max_delta_width = 64
    vertical_margin = 0
    horizontal_margin = 0
    #num_classes = 50


if __name__ == "__main__":

    # load data
    print("Load data from path...")
    train_data_fnames, train_labels = load_data_filename(hp.training_file)
    test_data_fnames, test_labels = load_data_filename(hp.testing_file)
    print(train_data_fnames.shape, test_data_fnames.shape)
    print(train_data_fnames[0])

    # Initialize distribute strategy
    strategy = tf.distribute.MirroredStrategy()

    # It seems that the auto share is not avalibel for our data type.
    # How ever it should be tf.data.experimental.AutoShardPolicy.FILE since the files >> workers.
    # If transfering data type to TFRecord, it will be good to FILE share policy.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    GLOBAL_BATCH_SIZE = hp.batch_size * strategy.num_replicas_in_sync

    print("Prefetching...")
    # Map function should update to TFRecord
    # instead of tf.py_function for better performance.
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data_fnames))
    train_dataset = train_dataset.map(lambda x: tf.py_function(load_npy, inp=[x], Tout=tf.float32),
                                      num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(GLOBAL_BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.with_options(options)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_data_fnames))
    test_dataset = test_dataset.map(lambda x: tf.py_function(load_npy, inp=[x], Tout=tf.float32),
                                    num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(GLOBAL_BATCH_SIZE)
    test_dataset = test_dataset.with_options(options)

    # print(f'training list\'s shape:{train_data.shape}, testing list\'s shape: {test_data.shape}')

    # to distributed strategy
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

    # Create Tensorboard Writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = op.join(hp.logdir, current_time)
    file_summary_writer = tf.summary.create_file_writer(log_dir)

    # whole training data
    if hp.steps_per_epoch == -1:
            steps_per_epoch = math.ceil(train_data_fnames.shape[0] / GLOBAL_BATCH_SIZE)
    elif hp.steps_per_epoch > 0:
            steps_per_epoch = int(hp.steps_per_epoch)
    else:
        raise ValueError(f"Wrong number assigned to steps_per_epoch: {hp.steps_per_epoch}")

    with strategy.scope():
        print("Build model...")
        model, optimizer, loss_fn = auto_encoder(hp.image_height, hp.image_width)
        print(model.summary())
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        # Restore if the path given
        if hp.checkpoint_restore_dir != '':
            checkpoint.restore(tf.train.latest_checkpoint(hp.checkpoint_restore_dir))

        # Would combine with loss_fn
        test_loss = tf.keras.metrics.Mean(name='test_loss')

        train_accuracy = tf.keras.metrics.MeanAbsoluteError(name='train_MAE_loss')
        test_accuracy = tf.keras.metrics.MeanAbsoluteError(name='test_MAE_loss')

        def compute_loss(ref, predictions):
            per_example_loss = loss_fn(ref, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)


        def create_mask():
            bbox = random_bbox(hp)
            regular_mask = bbox2mask(hp, bbox)
            mask = tf.cast(regular_mask, tf.float32)
            return mask

        def train_step(inputs):
            y = inputs
            x = inputs

            mask = create_mask()
            x_incomplete = x * (1.-mask)

            ones_x = tf.ones_like(x_incomplete)
            model_input = tf.concat([x_incomplete, ones_x*mask], axis=3)

            with tf.GradientTape() as tape:
                predictions = model(inputs=model_input, training=True)
                loss = compute_loss(y, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # This is counted without GLOBAL_BATCH_SIZE
            train_accuracy.update_state(y, predictions)

            summary_images = [x_incomplete, predictions, y]
            summary_images = tf.concat(summary_images, axis=2)
            return loss, summary_images

        def test_step(inputs):
            y = inputs
            x = inputs

            mask = create_mask()
            x_incomplete = x * (1.-mask)
            ones_x = tf.ones_like(x_incomplete)
            model_input = tf.concat([x_incomplete, ones_x*mask], axis=3)

            predictions = model(model_input, training=False)
            t_loss = loss_fn(y, predictions)

            test_loss.update_state(t_loss)
            test_accuracy.update_state(y, predictions)

        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses, summary_images = strategy.run(train_step, args=(dataset_inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None), summary_images

        @tf.function
        def distributed_test_step(dataset_inputs):
            return strategy.run(test_step, args=(dataset_inputs,))

        # Experiment Epoch
        for epoch in range(hp.epochs):
            total_loss = 0.0
            num_batches = 0
            summary_images_list = []

            ### Training loop.
            train_iter = iter(train_dist_dataset)
            for batch_step in tqdm(range(steps_per_epoch)):
                # The step here refers to whole batch
                step_loss, summary_images = distributed_train_step(next(train_iter))
                total_loss += step_loss
                num_batches += 1

                # Cast tensor from Replica
                if strategy.num_replicas_in_sync > 1:
                    summary_images = summary_images.values
                    # concat on batch channel: n * [b, h, w, c] -> b*n, h, w, c
                    summary_images = tf.concat(summary_images, axis=0)
                summary_images_list.append(summary_images)

                if hp.profile:
                    if epoch == 0 and batch_step == 9:
                        tf.profiler.experimental.start(hp.logdir)
                    if epoch == 0 and batch_step == 19:
                        tf.profiler.experimental.stop()

            # concat all sample on batch channel
            summary_images_list = tf.concat(summary_images_list, axis=0)
            train_loss = total_loss / num_batches

            ### Testing loop
            for x in test_dist_dataset:
                distributed_test_step(x)

            # Checkpoint save.
            if epoch % hp.checkpoint_freq == 0:
                checkpoint.save(hp.checkpoint_prefix)

            # Write to tensorboard.
            with file_summary_writer.as_default():
                scalar_summary('train loss', train_loss, step=epoch)
                scalar_summary('train MAE loss', train_accuracy.result(), step=epoch)
                images_summary("Training result", summary_images_list, step=epoch, max_outputs=hp.max_outputs)

                scalar_summary('test loss', test_loss.result(), step=epoch)
                scalar_summary('test MAE loss', test_accuracy.result(), step=epoch)

            template = ("Epoch {}, Loss: {}, MAE loss: {}, Test Loss: {}, "
                        "Test MAE loss: {}")
            print(template.format(epoch+1, train_loss,
                                  train_accuracy.result()*100, test_loss.result(),
                                  test_accuracy.result()*100))

            test_loss.reset_states()
            train_accuracy.reset_states()
            test_accuracy.reset_states()

