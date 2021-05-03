from data_loader import load_npy, load_data_filename
from summary_ops import scalar_summary, images_summary
from summary_ops import dict_scalar_summary
from summary_ops import gradient_calc
from inpaint_ops import random_bbox, bbox2mask, gan_hinge_loss
from inpaint_ops import brush_stroke_mask
from inpaint_ops import mag_mel_weighted_map
from tqdm import tqdm
from pretrain.models import coarse_inpaint_net
from SPADE import image_encoder, generator, discriminator
from sp_ops import generator_loss, discriminator_loss
from sp_ops import feature_loss, kl_loss, L1_loss
from sn_layers import SNConv2D

import tensorflow as tf
import os.path as op
import datetime
import math

print(tf.__version__)


class hp:
    # Training setting
    data_file = 'esc50_mag'
    labeled = False  # 15:True, large:False
    save_descript = '_spadeNet_GPU8'
    debug_graph = False
    training_file = op.join('./data', data_file, 'train_list.txt')
    testing_file = op.join('./data', data_file, 'test_list.txt')
    logdir = op.join('./logs', f'{data_file}{save_descript}')
    # checkpoint_dir = op.join('./checkpoints', data_file)
    pretrain_model = 'pretrain_models/first_stage'
    checkpoint_prefix = op.join(logdir, "ckpt")
    checkpoint_restore_dir = ''
    checkpoint_freq = 100
    restore_epochs = 0  # Specify for restore training.
    epochs = 10000
    steps_per_epoch = -1  # -1: whole training data.
    batch_size = 16
    max_outputs = 5
    profile = False  # profile on first epoch, batch 10~20.
    l1_alpha = 1.
    weighted_loss = True
    gan_alpha = 1.
    feature_alpha = 10.
    kl_alpha = 0.05
    kl_sim_alpha = 0.05
    # Data
    image_height = 256
    image_width = 256
    image_channel = 1
    length_5sec = 862
    mask_height = 256
    mask_width = round(length_5sec * 0.2 * 0.35)        # max of mask width
    max_delta_height = 0
    max_delta_width = round(length_5sec * 0.2 * 0.1)    # decrease with this delta
    vertical_margin = 0
    horizontal_margin = 0  # match with deepfill
    ir_mask = False
    # num_classes = 50


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
    train_dataset = train_dataset.shuffle(buffer_size=len(train_data_fnames))
    train_dataset = train_dataset.map(lambda x: tf.numpy_function(load_npy, inp=[x], Tout=tf.float32),
                                      num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(GLOBAL_BATCH_SIZE)
    # train_dataset = train_dataset.shuffle(buffer_size=len(train_data_fnames)).batch(GLOBAL_BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.with_options(options)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_data_fnames))
    test_dataset = test_dataset.map(lambda x: tf.numpy_function(load_npy, inp=[x], Tout=tf.float32),
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
        # build first model, load pretrain weight, freeze weights.
        # first_stage, optimizer = coarse_inpaint_net(hp.image_height, hp.image_width, hp.image_channel)
        # first_ckpt = tf.train.Checkpoint(
        #     optimizer=optimizer,
        #     model=first_stage)
        # first_ckpt.restore(tf.train.latest_checkpoint(hp.pretrain_weights_dir))
        first_stage = tf.keras.models.load_model(hp.pretrain_model)
        print(first_stage.summary())
        for layer in first_stage.layers:
            layer.trainable = False

        # TTUR
        g_lr = tf.keras.optimizers.schedules.InverseTimeDecay(
            1e-4,
            decay_steps=steps_per_epoch*1000,
            decay_rate=1,
            staircase=False)
        d_lr = tf.keras.optimizers.schedules.InverseTimeDecay(
            2e-4,
            decay_steps=steps_per_epoch*1000,
            decay_rate=1,
            staircase=False)
        beta_1 = 0.0
        beta_2 = 0.9

        encoder = image_encoder(hp.image_height, hp.image_width, hp.image_channel)
        generator = generator(hp.image_height, hp.image_width, hp.image_channel, hp.batch_size)
        g_optimizer = tf.keras.optimizers.Adam(g_lr, beta_1=beta_1, beta_2=beta_2)
        print(encoder.summary())
        print(generator.summary())

        discriminator = discriminator(hp.image_height, hp.image_width, hp.image_channel)
        d_optimizer = tf.keras.optimizers.Adam(d_lr, beta_1=beta_1, beta_2=beta_2)
        print(discriminator.summary())

        checkpoint = tf.train.Checkpoint(
                        generator_optimizer=g_optimizer,
                        discriminator_optimizer=d_optimizer,
                        encoder=encoder,
                        generator=generator,
                        discriminator=discriminator)
        ckpt_manager = tf.train.CheckpointManager(
            checkpoint,
            directory=hp.logdir,
            checkpoint_name='ckpt',
            max_to_keep=10
        )
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
            batch_factor = hp.batch_size / GLOBAL_BATCH_SIZE
            L = loss * batch_factor
            return L
            # per_example_loss = tf.math.abs(ref - predictions)
            # sample_weight in method works like attention mapt
            # return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

        def create_mask():
            bbox = random_bbox(hp)
            regular_mask = bbox2mask(hp, bbox)
            mask = tf.cast(regular_mask, tf.float32)
            if hp.ir_mask:
                ir_mask = brush_stroke_mask(hp)
                mask = tf.cast(
                    tf.logical_or(
                        tf.cast(ir_mask, tf.bool),
                        tf.cast(mask, tf.bool)
                    ),
                    tf.float32
                )
            return mask

        @tf.function
        def train_step(inputs):
            x_pos, mask = inputs
            loss = {}
            # First Stage.
            x_incomplete = x_pos * (1.-mask)
            first_stage_input = [x_incomplete, mask]
            x_stage1 = first_stage(inputs=first_stage_input, training=False)

            g_1st_diff = tf.math.abs(x_pos - x_stage1)
            if hp.weighted_loss:
                g_1st_diff = mag_mel_weighted_map(g_1st_diff)
            g_1st_loss = compute_global_loss(tf.math.reduce_mean(g_1st_diff))
            # First stage incomplete
            semap = x_pos * (1.-mask) + x_stage1 * mask

            with tf.GradientTape(persistent=True) as tape:
                # Encoder
                encoder_input = tf.concat([x_pos, semap], axis=0)
                x_mean, x_var = encoder(inputs=encoder_input, training=True)
                x_pos_mean, semap_mean = tf.split(x_mean, 2)
                x_pos_var, semap_var = tf.split(x_var, 2)
                # Generator
                G_input = [semap, x_pos_mean, x_pos_var]
                x_stage2 = generator(inputs=G_input, training=True)
                x_complete = x_pos * (1.-mask) + x_stage2 * mask
                # Discriminator
                semap_din = tf.concat([semap, semap], axis=0)
                x_din = tf.concat([x_pos, x_stage2], axis=0)
                D_input = [semap_din, x_din]

                pos_neg = discriminator(inputs=D_input, training=True)
                pos = []
                neg = []
                for i_scale in pos_neg:
                    pos_logits = []
                    neg_logits = []
                    for j_logits in i_scale:
                        j_pos, j_neg = tf.split(j_logits, 2)
                        pos_logits.append(j_pos)
                        neg_logits.append(j_neg)
                    pos.append(pos_logits)
                    neg.append(neg_logits)
                # pos, neg = tf.split(pos_neg, 2)

                # Calc loss
                g_adv_loss = hp.gan_alpha * generator_loss(neg)
                g_adv_loss = compute_global_loss(g_adv_loss)

                g_kl_loss = hp.kl_alpha * kl_loss(x_pos_mean, x_pos_var)
                g_kl_loss = compute_global_loss(g_kl_loss)
                # We dont have pretrain weights like imagenet in audio.
                # g_vgg_loss = hp.vgg_alpha * VGGLoss()(x_pos, x_stage2)
                g_feature_loss = hp.feature_alpha * feature_loss(pos, neg)
                g_feature_loss = compute_global_loss(g_feature_loss)

                mean_loss = L1_loss(x_pos_mean, semap_mean)
                mean_loss = compute_global_loss(mean_loss)
                var_loss = L1_loss(x_pos_var, semap_var)
                var_loss = compute_global_loss(var_loss)
                g_sim_loss = hp.kl_sim_alpha * (mean_loss + var_loss)

                g_2st_diff = tf.math.abs(x_pos - x_stage2)
                if hp.weighted_loss:
                    g_2st_diff = mag_mel_weighted_map(g_2st_diff)
                g_2st_loss = hp.l1_alpha * tf.math.reduce_mean(g_2st_diff)
                g_2st_loss = compute_global_loss(g_2st_loss)
                # Reg loos?

                d_adv_loss, d_real, d_fake = discriminator_loss(pos, neg)
                d_adv_loss = hp.gan_alpha * d_adv_loss
                d_adv_loss = compute_global_loss(d_adv_loss)
                d_real = compute_global_loss(d_real)
                d_fake = compute_global_loss(d_fake)
                # Reg loss?

                g_loss = g_adv_loss + g_kl_loss + g_feature_loss + g_2st_loss + g_sim_loss
                d_loss = d_adv_loss

            # Concat list of vars into one list.
            g_vars = encoder.trainable_variables + generator.trainable_variables

            g_gradients = tape.gradient(g_loss, g_vars)
            d_gradients = tape.gradient(d_loss, discriminator.trainable_variables)

            g_optimizer.apply_gradients(zip(g_gradients, g_vars))
            d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

            loss['s1_loss'] = g_1st_loss
            loss['s2_loss'] = g_2st_loss
            loss['d_loss'] = d_loss
            loss['d_real'] = d_real
            loss['d_fake'] = d_fake
            loss['g_adv_loss'] = g_adv_loss
            loss['g_loss'] = g_loss
            loss['kl_loss'] = g_kl_loss
            loss['sim_loss'] = g_sim_loss
            loss['kl_mean'] = mean_loss
            loss['kl_var'] = var_loss
            loss['feature_loss'] = g_feature_loss
            loss['g_adv_to_x2'] = gradient_calc(g_adv_loss, x_stage2)
            # loss['g_l1_to_x2'] = gradient_calc(g_2st_loss, x_stage2)
            loss['g_feature_to_x2'] = gradient_calc(g_feature_loss, x_stage2)
            # loss['g_kl_to_x2'] = gradient_calc(g_kl_loss, x_stage2)
            # loss['g_sim_to_x2'] = gradient_calc(g_sim_loss, x_stage2)

            summary_images = [x_incomplete, x_stage1, x_stage2, x_complete, x_pos, semap]
            summary_images = tf.concat(summary_images, axis=2)

            train_accuracy.update_state(x_pos, x_complete)

            return loss, summary_images

        def test_step(inputs):
            x_pos, mask = inputs

            x_incomplete = x_pos * (1.-mask)
            first_stage_input = [x_incomplete, mask]
            x_stage1 = first_stage(inputs=first_stage_input, training=False)
            semap = x_incomplete + x_stage1 * mask
            x_mean, x_var = encoder(inputs=semap, training=False)
            G_input = [semap, x_mean, x_var]
            x_stage2 = generator(inputs=G_input, training=False)
            x_complete = x_incomplete + x_stage2 * mask
            t_loss = tf.math.reduce_mean(tf.math.abs(x_pos - x_stage2))
            # t_loss = loss_fn(x_pos, x_stage2)
            test_loss.update_state(t_loss)
            test_accuracy.update_state(x_pos, x_complete)

            summary_images = [x_incomplete, x_stage1, x_stage2, x_complete, x_pos, semap]
            summary_images = tf.concat(summary_images, axis=2)

            return summary_images

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
            # total_loss = 0.0
            num_batches = 0
            summary_images_list = []

            # Training loop.
            train_iter = iter(train_dist_dataset)
            for batch_step in tqdm(range(steps_per_epoch)):
                # The step here refers to whole batch
                step = epoch * steps_per_epoch + batch_step

                mask = create_mask()
                x_pos = next(train_iter)
                step_loss, summary_images = distributed_train_step([x_pos, mask])

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
                    scalar_summary('G_lr', g_optimizer._decayed_lr(tf.float32), step=step)
                    scalar_summary('D_lr', d_optimizer._decayed_lr(tf.float32), step=step)
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
                mask = create_mask()
                x_pos = next(test_iter)
                summary_images = distributed_test_step([x_pos, mask])

                if strategy.num_replicas_in_sync > 1:
                    summary_images = summary_images.values
                    summary_images = tf.concat(summary_images, axis=0)
                with file_summary_writer.as_default():
                    images_summary("Testing result", summary_images, step=step, max_outputs=hp.max_outputs)

            # Checkpoint save.
            if epoch % hp.checkpoint_freq == 0:
                #checkpoint.save(hp.checkpoint_prefix)
                ckpt_manager.save()

            # Write to tensorboard.
            with file_summary_writer.as_default():
                dict_scalar_summary('train loss', total_loss, step=epoch)
                scalar_summary('train MAE loss', train_accuracy.result(), step=epoch)
                #images_summary("Training result", summary_images_list, step=epoch, max_outputs=hp.max_outputs)

                scalar_summary('test loss', test_loss.result(), step=epoch)
                scalar_summary('test MAE loss', test_accuracy.result(), step=epoch)

            template = ("Epoch {}, Loss: {}, MAE loss: {}, Test Loss: {}, "
                        "Test MAE loss: {}")
            print(template.format(epoch+1, total_loss['s2_loss'],
                                  train_accuracy.result()*100, test_loss.result(),
                                  test_accuracy.result()*100))

            test_loss.reset_states()
            train_accuracy.reset_states()
            test_accuracy.reset_states()
