from data_loader import load_npy, load_data_filename
from models import inpaint_net, sn_patch_gan_discriminator
from summary_ops import scalar_summary, images_summary
from summary_ops import dict_scalar_summary
from summary_ops import gradient_calc
from inpaint_ops import random_bbox, bbox2mask, gan_hinge_loss
from tqdm import tqdm

import tensorflow as tf
import os.path as op
import datetime
import math

print(tf.__version__)


class hp:
    # Training setting
    data_file = 'spec_15'
    labeled = True  # 15:True, large:False
    save_descript = '_net_wDmySNrefbigG_woCA_woFlow_testD'
    debug_graph = False
    training_file = op.join('./data', data_file, 'train_list.txt')
    testing_file = op.join('./data', data_file, 'test_list.txt')
    logdir = op.join('./logs_debug', f'{data_file}{save_descript}')
    #checkpoint_dir = op.join('./checkpoints', data_file)
    checkpoint_prefix = op.join(logdir, "ckpt")
    checkpoint_restore_dir = ''
    checkpoint_freq = 10
    restore_epochs = 0  # Specify for restore training.
    epochs = 200
    steps_per_epoch = -1  # -1: whole training data.
    validation_split = 0.1
    batch = True
    batch_size = 32
    max_outputs = 5
    profile = False  # profile on first epoch, batch 10~20.
    l1_loss_alpha = 1.
    gan_loss_alpha = 1.
    # Data
    image_height = 256
    image_width = 256
    image_channel = 1
    mask_height = 256
    mask_width = 64
    max_delta_height = 0
    max_delta_width = 32
    vertical_margin = 0
    horizontal_margin = 65  # match with deepfill
    #num_classes = 50


if __name__ == "__main__":

    # load data
    print("Load data from path...")
    train_data_fnames, train_labels = load_data_filename(hp.training_file, hp.labeled)
    test_data_fnames, test_labels = load_data_filename(hp.testing_file, hp.labeled)
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
    # Skip the last batches for the batch size consistency, due to error might occur in CA module.
    steps_per_epoch -= 1

    # whole training data
    if hp.steps_per_epoch == -1:
            test_steps_per_epoch = math.ceil(test_data_fnames.shape[0] / GLOBAL_BATCH_SIZE)
    elif hp.steps_per_epoch > 0:
            test_steps_per_epoch = int(hp.steps_per_epoch)
    else:
        raise ValueError(f"Wrong number assigned to steps_per_epoch: {hp.steps_per_epoch}")
    # Skip the last batches for the batch size consistency, due to error might occur in CA module.
    test_steps_per_epoch -= 1


    with strategy.scope():
        print("Build model...")
        g_model = inpaint_net(hp.image_height, hp.image_width, hp.batch_size)
        g_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.999)
        loss_fn = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
        print(g_model.summary())
        d_model = sn_patch_gan_discriminator(hp.image_height, hp.image_width)
        d_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.999)
        print(d_model.summary())

        checkpoint = tf.train.Checkpoint(
                        generator_optimizer=g_optimizer,
                        discriminator_optimizer=d_optimizer,
                        generator=g_model,
                        discriminator=d_model)
        # Restore if the path given
        if hp.checkpoint_restore_dir != '':
            checkpoint.restore(tf.train.latest_checkpoint(hp.checkpoint_restore_dir))

        # Would combine with loss_fn
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        train_accuracy = tf.keras.metrics.MeanAbsoluteError(name='train_MAE_loss')
        test_accuracy = tf.keras.metrics.MeanAbsoluteError(name='test_MAE_loss')


        # The function here only consider the first dimension(B) of losses,
        # wont mean over dims: H, W, C, due to reduce_sup in methods.
        # Calculate the distributed losses by myself.
        def compute_global_loss(loss):
            batch_div = hp.batch_size / GLOBAL_BATCH_SIZE
            L = loss * batch_div
            return L
            #per_example_loss = tf.math.abs(ref - predictions)
            # sample_weight in method works like attention mapt
            # return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

        def create_mask():
            bbox = random_bbox(hp)
            regular_mask = bbox2mask(hp, bbox)
            mask = tf.cast(regular_mask, tf.float32)
            return mask

        @tf.function
        def train_step(inputs):
            #x_pos = inputs
            x_pos, mask = inputs
            loss = {}

            #mask = create_mask()
            x_incomplete = x_pos * (1.-mask)

            model_input = [x_incomplete, mask]

            #model_input = x_pos

            #with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            with tf.GradientTape(persistent=True) as tape:
                # Generator
                #x_stage1, x_stage2 = g_model(inputs=model_input, training=True)
                #x_stage1, x_stage2, x_s_in, offset_flow = g_model(inputs=model_input, training=True)
                x_stage1, x_stage2, x_s_in = g_model(inputs=model_input, training=True)

                x_predicted = x_stage2
                x_complete = x_predicted * mask + x_incomplete * (1.-mask)
                #x_complete = x_stage2

                # l1 loss
                s1_loss = compute_global_loss(tf.math.reduce_mean(tf.math.abs(x_pos - x_stage1)))
                s2_loss = compute_global_loss(tf.math.reduce_mean(tf.math.abs(x_pos - x_stage2)))
                ae_loss = s1_loss + s2_loss

                # Discriminator

                x_pos_neg = tf.concat([x_pos, x_complete], axis=0)
                x_pos_neg_shape = x_pos_neg.get_shape().as_list()
                x_pos_neg = tf.concat([x_pos_neg,
                                       tf.tile(mask, [x_pos_neg_shape[0], 1, 1, 1])], axis=3)
                pos_neg = d_model(inputs=x_pos_neg, training=True)

                pos, neg = tf.split(pos_neg, 2)
                g_loss_0, d_loss, hp_loss, hn_loss = gan_hinge_loss(pos, neg)
                hp_loss = compute_global_loss(hp_loss)
                hn_loss = compute_global_loss(hn_loss)
                g_loss_0 = compute_global_loss(g_loss_0)
                d_loss = compute_global_loss(d_loss)

                g_loss = hp.l1_loss_alpha * ae_loss + hp.gan_loss_alpha * g_loss_0

            g_gradients = tape.gradient(g_loss, g_model.trainable_variables)
            d_gradients = tape.gradient(d_loss, d_model.trainable_variables)

            g_optimizer.apply_gradients(zip(g_gradients, g_model.trainable_variables))
            d_optimizer.apply_gradients(zip(d_gradients, d_model.trainable_variables))

            loss['s1_loss'] = s1_loss
            loss['s2_loss'] = s2_loss
            loss['ae_loss'] = ae_loss
            loss['d_loss'] = d_loss
            loss['g_loss'] = g_loss_0
            loss['hinge_pos_loss'] = hp_loss
            loss['hinge_neg_loss'] = hn_loss
            loss['g_to_x2'] = gradient_calc(g_loss_0, x_stage2)
            loss['ae_to_x1'] = gradient_calc(ae_loss, x_stage1)
            loss['ae_to_x2'] = gradient_calc(ae_loss, x_stage2)

            #summary_images = [x_incomplete, x_stage1, x_stage2, x_pos, x_s_in, offset_flow]
            summary_images = [x_incomplete, x_stage1, x_stage2, x_complete, x_pos, x_s_in]
            #summary_images = [x_incomplete, x_stage1, x_stage2, x_pos, x_s_in]
            #summary_images = [x_stage1, x_stage2, x_pos, x_s_in]
            summary_images = tf.concat(summary_images, axis=2)

            train_accuracy.update_state(x_pos, x_complete)

            return loss, summary_images

        def test_step(inputs):
            x_pos = inputs

            mask = create_mask()
            x_incomplete = x_pos * (1.-mask)
            model_input = [x_incomplete, mask]

            x_stage1, x_stage2, _ = g_model(model_input, training=False)
            x_complete = x_stage2 * mask + x_incomplete * (1.-mask)
            t_loss = loss_fn(x_pos, x_stage1)
            t_loss += loss_fn(x_pos, x_stage2)

            test_loss.update_state(t_loss)
            test_accuracy.update_state(x_pos, x_complete)

        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses, summary_images = strategy.run(train_step, args=(dataset_inputs,))
            for key in per_replica_losses:
                per_replica_losses[key] = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[key], axis=None)
            return per_replica_losses, summary_images

        @tf.function
        def distributed_test_step(dataset_inputs):
            return strategy.run(test_step, args=(dataset_inputs,))


        if hp.debug_graph:
            inputs = tf.zeros([hp.batch_size, 256, 256, 1])
            tf.summary.trace_on(graph=True, profiler=False)
            _ = train_step(inputs, hp.debug_graph)
            with file_summary_writer.as_default():
                tf.summary.trace_export(
                    name="train_step_trace",
                    step=0,
                    profiler_outdir=hp.logdir
                )

        # Experiment Epoch
        for epoch in range(hp.restore_epochs, hp.epochs+hp.restore_epochs):
            total_loss = {}
            #total_loss = 0.0
            num_batches = 0
            summary_images_list = []

            ### Training loop.
            train_iter = iter(train_dist_dataset)
            for batch_step in tqdm(range(steps_per_epoch)):
                # The step here refers to whole batch
                step = epoch * steps_per_epoch + batch_step

                mask = create_mask()
                x_pos = next(train_iter)
                step_loss, summary_images = distributed_train_step([x_pos, mask])
                #step_loss, summary_images = distributed_train_step(next(train_iter))

                for key in step_loss:
                    if key in total_loss:
                        total_loss[key] += step_loss[key]
                    else:
                        total_loss[key] = step_loss[key]
                # total_loss += step_loss

                num_batches += 1
                '''
                # Cast tensor from Replica
                if batch_step == 0: # Only collect first batch due to OOM
                    if strategy.num_replicas_in_sync > 1:
                        summary_images = summary_images.values
                        # concat on batch channel: n * [b, h, w, c] -> b*n, h, w, c
                        summary_images = tf.concat(summary_images, axis=0)

                    summary_images_list.append(summary_images)
                '''

                if strategy.num_replicas_in_sync > 1:
                    summary_images = summary_images.values
                    summary_images = tf.concat(summary_images, axis=0)

                step = epoch * steps_per_epoch + batch_step
                with file_summary_writer.as_default():
                    dict_scalar_summary('train loss step', step_loss, step=step)
                    scalar_summary('train MAE loss', train_accuracy.result(), step=step)
                    images_summary("Training result", summary_images, step=step, max_outputs=hp.max_outputs)

                if hp.profile:
                    if epoch == 0 and batch_step == 9:
                        tf.profiler.experimental.start(hp.logdir)
                    if epoch == 0 and batch_step == 14:
                        tf.profiler.experimental.stop()

            # concat all sample on batch channel
            #summary_images_list = tf.concat(summary_images_list, axis=0)
            # total_loss = total_loss / num_batches
            for key in total_loss:
                total_loss[key] = total_loss[key] / num_batches

            ### Testing loop
            test_iter = iter(test_dist_dataset)
            for batch_step in tqdm(range(test_steps_per_epoch)):
                distributed_test_step(next(test_iter))

            # Checkpoint save.
            if epoch % hp.checkpoint_freq == 0:
                checkpoint.save(hp.checkpoint_prefix)


            # Write to tensorboard.
            with file_summary_writer.as_default():
                dict_scalar_summary('train loss', total_loss, step=epoch)
                scalar_summary('train MAE loss', train_accuracy.result(), step=epoch)
                #images_summary("Training result", summary_images_list, step=epoch, max_outputs=hp.max_outputs)

                scalar_summary('test loss', test_loss.result(), step=epoch)
                scalar_summary('test MAE loss', test_accuracy.result(), step=epoch)

            template = ("Epoch {}, Loss: {}, MAE loss: {}, Test Loss: {}, "
                        "Test MAE loss: {}")
            print(template.format(epoch+1, total_loss['ae_loss'],
                                  train_accuracy.result()*100, test_loss.result(),
                                  test_accuracy.result()*100))

            test_loss.reset_states()
            train_accuracy.reset_states()
            test_accuracy.reset_states()
