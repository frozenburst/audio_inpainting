from data_loader import load_npy, load_data_filename
from summary_ops import scalar_summary, images_summary
from summary_ops import dict_scalar_summary
from summary_ops import gradient_calc
from summary_ops import audio_summary
from inpaint_ops import random_bbox, bbox2mask
from inpaint_ops import brush_stroke_mask
from tqdm import tqdm
from SPADE import image_encoder, generator, discriminator
from sp_ops import generator_loss, discriminator_loss
from sp_ops import feature_loss, kl_loss, L1_loss, stft_loss

from libs.mb_melgan.configs.mb_melgan import MultiBandMelGANGeneratorConfig
from libs.mb_melgan.models.mb_melgan import TFPQMF, TFMelGANGenerator
from utils import mag_to_mel

import tensorflow as tf
import os.path as op
import datetime
import math
import yaml

print(tf.__version__)


class hp:
    # Training setting
    data_file = 'maestro'   # esc50, maestro, ljs
    isMag = True
    labeled = False  # 15:True, large:False
    save_descript = '_spadeNet_bs16_AllWeighted_Novocol_m10_110'
    debug_graph = False
    training_file = op.join('./data', data_file, 'train_list.txt')
    testing_file = op.join('./data', data_file, 'test_list.txt')
    logdir = op.join('./logs', f'{data_file}{save_descript}')
    checkpoint_prefix = op.join(logdir, "ckpt")
    checkpoint_restore_dir = './logs/maestro_spadeNet_bs16_AllWeighted_Novocol_m10_110'
    checkpoint_freq = 100
    restore_epochs = 900  # Specify for restore training.
    epochs = 10000
    summary_freq = 50
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
    vocol_loss = False
    stft_alpha = 10.    # Serve as perceptual
    # Data
    sr = 44100          # ljs: 22050, others: 44100
    hop_size = 256
    image_height = 256
    image_width = 256
    image_channel = 1
    length_5sec = int((sr / hop_size) * 5)              # int() = floor()
    mask_height = 256
    mask_width = round(length_5sec * 0.2 * 1.1)        # max of mask width
    max_delta_height = 0
    max_delta_width = round(length_5sec * 0.2 * 1.0)    # decrease with this delta
    vertical_margin = 0
    horizontal_margin = 0
    ir_mask = False
    # Vocoder
    v_ckpt = f'libs/mb_melgan/ckpt/{data_file}/generator-800000.h5'
    v_config = f'libs/mb_melgan/configs/multiband_melgan.{data_file}_v1.yaml'


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
    # However it should be tf.data.experimental.AutoShardPolicy.FILE since the files >> workers.
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
    train_dataset = train_dataset.map(lambda x: tf.numpy_function(load_npy, inp=[x, hp.length_5sec], Tout=tf.float32),
                                      num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.with_options(options)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_data_fnames))
    test_dataset = test_dataset.map(lambda x: tf.numpy_function(load_npy, inp=[x, hp.length_5sec], Tout=tf.float32),
                                    num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
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
        with open(hp.v_config) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        mb_melgan = TFMelGANGenerator(
            config=MultiBandMelGANGeneratorConfig(**config["multiband_melgan_generator_params"]),
            name="multiband_melgan_generator",)
        mb_melgan._build()
        mb_melgan.load_weights(hp.v_ckpt)
        pqmf = TFPQMF(
            config=MultiBandMelGANGeneratorConfig(**config["multiband_melgan_generator_params"]), name="pqmf")
        print(mb_melgan.summary())

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
        # Restore if the path given
        if hp.checkpoint_restore_dir != '':
            checkpoint.restore(tf.train.latest_checkpoint(hp.checkpoint_restore_dir))

        ckpt_manager = tf.train.CheckpointManager(
            checkpoint,
            directory=hp.logdir,
            checkpoint_name='ckpt',
            max_to_keep=10
        )

        # Would combine with loss_fn
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_weighted_loss = tf.keras.metrics.Mean(name='test_weighted_loss')
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
            x_ori, mask = inputs
            rands = tf.experimental.numpy.random.randint(0, hp.length_5sec-hp.image_width, dtype=tf.int32)
            x_pos = x_ori[:, :, rands:rands+hp.image_width, :]
            x_pos.set_shape([hp.batch_size, hp.image_height, hp.image_width, hp.image_channel])

            loss = {}
            x_incomplete = x_pos * (1.-mask)

            with tf.GradientTape(persistent=True) as tape:
                semap = x_incomplete

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
                # Vocoder
                pre = x_ori[:, :, :rands, :]
                post = x_ori[:, :, rands+x_incomplete.shape[-2]:, :]
                incomplete = tf.concat([pre, x_incomplete, post], axis=2)
                complete = tf.concat([pre, x_complete, post], axis=2)
                incomplete.set_shape(x_ori.shape)
                complete.set_shape(x_ori.shape)

                if hp.isMag:
                    pos_mels = mag_to_mel(x_ori, hp.sr)
                    incomplete_mels = mag_to_mel(incomplete, hp.sr)
                    x_mels = mag_to_mel(complete, hp.sr)
                else:
                    pos_mels = x_ori
                    incomplete_mels = incomplete
                    x_mels = complete

                x_subbands = mb_melgan(x_mels, training=False)
                x_audios = pqmf.synthesis(x_subbands)

                pos_subbands = mb_melgan(pos_mels, training=False)
                pos_audios = pqmf.synthesis(pos_subbands)

                incomplete_subbands = mb_melgan(incomplete_mels, training=False)
                incomplete_audios = pqmf.synthesis(incomplete_subbands)

                # Calc loss
                if hp.vocol_loss:
                    g_vocol_loss = hp.stft_alpha * stft_loss(x_audios, pos_audios, hp.weighted_loss)
                    g_vocol_loss = compute_global_loss(g_vocol_loss)

                g_adv_loss = hp.gan_alpha * generator_loss(neg, hp.weighted_loss)
                g_adv_loss = compute_global_loss(g_adv_loss)

                g_kl_loss = hp.kl_alpha * kl_loss(x_pos_mean, x_pos_var)
                g_kl_loss = compute_global_loss(g_kl_loss)
                # We dont have pretrain weights like imagenet in audio.
                # g_vgg_loss = hp.vgg_alpha * VGGLoss()(x_pos, x_stage2)
                g_feature_loss = hp.feature_alpha * feature_loss(pos, neg, hp.weighted_loss)
                g_feature_loss = compute_global_loss(g_feature_loss)

                mean_loss = L1_loss(x_pos_mean, semap_mean, hp.weighted_loss)
                mean_loss = compute_global_loss(mean_loss)
                var_loss = L1_loss(x_pos_var, semap_var)
                var_loss = compute_global_loss(var_loss)
                g_sim_loss = hp.kl_sim_alpha * (mean_loss + var_loss)

                g_2st_loss = hp.l1_alpha * L1_loss(x_pos, x_stage2, hp.weighted_loss)
                g_2st_loss = compute_global_loss(g_2st_loss)
                # Reg loos?

                d_adv_loss, d_real, d_fake = discriminator_loss(pos, neg, hp.weighted_loss)
                d_adv_loss = hp.gan_alpha * d_adv_loss
                d_adv_loss = compute_global_loss(d_adv_loss)
                d_real = compute_global_loss(d_real)
                d_fake = compute_global_loss(d_fake)
                # Reg loss?

                if hp.vocol_loss:
                    g_loss = g_adv_loss + g_kl_loss + g_feature_loss + g_2st_loss + g_sim_loss + g_vocol_loss
                else:
                    g_loss = g_adv_loss + g_kl_loss + g_feature_loss + g_2st_loss + g_sim_loss

                d_loss = d_adv_loss

            # Concat list of vars into one list.
            g_vars = encoder.trainable_variables + generator.trainable_variables

            g_gradients = tape.gradient(g_loss, g_vars)
            d_gradients = tape.gradient(d_loss, discriminator.trainable_variables)

            g_optimizer.apply_gradients(zip(g_gradients, g_vars))
            d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

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
            if hp.vocol_loss:
                loss['vocol_loss'] = g_vocol_loss
            loss['g_adv_to_x2'] = gradient_calc(g_adv_loss, x_stage2)
            loss['g_feature_to_x2'] = gradient_calc(g_feature_loss, x_stage2)

            summary_images = [x_incomplete, x_stage2, x_complete, x_pos]
            summary_images = tf.concat(summary_images, axis=2)

            summary_audios = tf.concat([x_audios, pos_audios, incomplete_audios], axis=2)

            train_accuracy.update_state(x_pos, x_complete)

            return loss, summary_images, summary_audios

        def test_step(inputs):
            x_ori, mask, = inputs
            rands = tf.experimental.numpy.random.randint(0, hp.length_5sec-hp.image_width, dtype=tf.int32)
            x_pos = x_ori[:, :, rands:rands+hp.image_width, :]
            x_pos.set_shape([hp.batch_size, hp.image_height, hp.image_width, hp.image_channel])

            x_incomplete = x_pos * (1.-mask)
            semap = x_incomplete
            x_mean, x_var = encoder(inputs=semap, training=False)
            G_input = [semap, x_mean, x_var]
            x_stage2 = generator(inputs=G_input, training=False)
            x_complete = x_incomplete + x_stage2 * mask

            # Vocoder
            pre = x_ori[:, :, :rands, :]
            post = x_ori[:, :, rands+x_incomplete.shape[-2]:, :]
            incomplete = tf.concat([pre, x_incomplete, post], axis=2)
            complete = tf.concat([pre, x_complete, post], axis=2)
            incomplete.set_shape(x_ori.shape)
            complete.set_shape(x_ori.shape)

            if hp.isMag:
                pos_mels = mag_to_mel(x_ori, hp.sr)
                incomplete_mels = mag_to_mel(incomplete, hp.sr)
                x_mels = mag_to_mel(complete, hp.sr)
            else:
                pos_mels = x_ori
                incomplete_mels = incomplete
                x_mels = complete

            x_subbands = mb_melgan(x_mels, training=False)
            x_audios = pqmf.synthesis(x_subbands)

            pos_subbands = mb_melgan(pos_mels, training=False)
            pos_audios = pqmf.synthesis(pos_subbands)

            incomplete_subbands = mb_melgan(incomplete_mels, training=False)
            incomplete_audios = pqmf.synthesis(incomplete_subbands)

            g_2st_loss = hp.l1_alpha * L1_loss(x_pos, x_stage2, True)
            g_2st_loss_noW = hp.l1_alpha * L1_loss(x_pos, x_stage2, False)

            test_loss.update_state(g_2st_loss_noW)
            test_weighted_loss.update_state(g_2st_loss)
            test_accuracy.update_state(x_pos, x_complete)

            summary_images = [x_incomplete, x_stage2, x_complete, x_pos]
            summary_images = tf.concat(summary_images, axis=2)

            summary_audios = tf.concat([x_audios, pos_audios, incomplete_audios], axis=2)

            return summary_images, summary_audios

        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses, summary_images, summary_audios = strategy.run(train_step, args=(dataset_inputs,))
            for key in per_replica_losses:
                per_replica_losses[key] = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[key], axis=None)
            return per_replica_losses, summary_images, summary_audios

        @tf.function
        def distributed_test_step(dataset_inputs):
            summary_images, summary_audios = strategy.run(test_step, args=(dataset_inputs,))
            return summary_images, summary_audios

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
            num_batches = 0
            summary_images_list = []

            # Training loop.
            train_iter = iter(train_dist_dataset)
            for batch_step in tqdm(range(steps_per_epoch)):
                # The step here refers to whole batch
                step = epoch * steps_per_epoch + batch_step

                mask = create_mask()
                x_ori = next(train_iter)

                step_loss, summary_images, summary_audios = distributed_train_step([x_ori, mask])

                for key in step_loss:
                    if key in total_loss:
                        total_loss[key] += step_loss[key]
                    else:
                        total_loss[key] = step_loss[key]

                num_batches += 1

                with file_summary_writer.as_default():
                    dict_scalar_summary('train loss step', step_loss, step=step)
                    scalar_summary('train MAE loss', train_accuracy.result(), step=step)
                    scalar_summary('G_lr', g_optimizer._decayed_lr(tf.float32), step=step)
                    scalar_summary('D_lr', d_optimizer._decayed_lr(tf.float32), step=step)

                if hp.profile:
                    if epoch == 0 and batch_step == 9:
                        tf.profiler.experimental.start(hp.logdir)
                    if epoch == 0 and batch_step == 14:
                        tf.profiler.experimental.stop()
            # Write freq lower to save storage.
            if epoch % hp.summary_freq == 0:
                if strategy.num_replicas_in_sync > 1:
                    summary_images = summary_images.values
                    summary_images = tf.concat(summary_images, axis=0)
                    summary_audios = summary_audios.values
                    summary_audios = tf.concat(summary_audios, axis=0)
                x_audios, pos_audios, incomplete_audios = tf.split(summary_audios, 3, axis=2)
                with file_summary_writer.as_default():
                    images_summary("Training result", summary_images, step=step, max_outputs=hp.max_outputs)
                    audio_summary("Training result/x_complete", x_audios, hp.sr, step=step, max_outputs=hp.max_outputs)
                    audio_summary("Training result/x_pos", pos_audios, hp.sr, step=step, max_outputs=hp.max_outputs)
                    audio_summary("Training result/x_incomplete", incomplete_audios, hp.sr, step=step, max_outputs=hp.max_outputs)

            # concat all sample on batch channel
            #summary_images_list = tf.concat(summary_images_list, axis=0)
            # total_loss = total_loss / num_batches
            for key in total_loss:
                total_loss[key] = total_loss[key] / num_batches

            ### Testing loop
            test_iter = iter(test_dist_dataset)
            for batch_step in tqdm(range(test_steps_per_epoch)):
                mask = create_mask()
                x_ori = next(test_iter)
                test_images, test_audios = distributed_test_step([x_ori, mask])

            if epoch % hp.summary_freq == 0:
                if strategy.num_replicas_in_sync > 1:
                    test_images = test_images.values
                    test_images = tf.concat(test_images, axis=0)

                    test_audios = test_audios.values
                    test_audios = tf.concat(test_audios, axis=0)
                test_x, test_pos, test_incomplete = tf.split(test_audios, 3, axis=2)

                with file_summary_writer.as_default():
                    images_summary("Testing result/images", test_images, step=step, max_outputs=hp.max_outputs)
                    audio_summary("Testing result/x_complete", test_x, hp.sr, step=step, max_outputs=hp.max_outputs)
                    audio_summary("Testing result/x_pos", test_pos, hp.sr, step=step, max_outputs=hp.max_outputs)
                    audio_summary("Testing result/x_incomplete", test_incomplete, hp.sr, step=step, max_outputs=hp.max_outputs)

            # Checkpoint save.
            if epoch % hp.checkpoint_freq == 0:
                #checkpoint.save(hp.checkpoint_prefix)
                ckpt_manager.save()

            # Write to tensorboard.
            with file_summary_writer.as_default():
                dict_scalar_summary('train loss', total_loss, step=epoch)
                scalar_summary('train MAE loss', train_accuracy.result(), step=epoch)

                scalar_summary('test loss', test_loss.result(), step=epoch)
                scalar_summary('test loss/weighted', test_weighted_loss.result(), step=epoch)
                scalar_summary('test MAE loss', test_accuracy.result(), step=epoch)

            template = ("Epoch {}, Loss: {}, MAE loss: {}, Test Loss: {}, "
                        "Test MAE loss: {}")
            print(template.format(epoch+1, total_loss['s2_loss'],
                                  train_accuracy.result()*100, test_loss.result(),
                                  test_accuracy.result()*100))

            test_loss.reset_states()
            test_weighted_loss.reset_states()
            train_accuracy.reset_states()
            test_accuracy.reset_states()
