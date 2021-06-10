from data_loader import load_npy_test, load_data_filename
from summary_ops import scalar_summary, images_summary
from summary_ops import dict_scalar_summary
from summary_ops import audio_summary
from tqdm import tqdm
from SPADE import image_encoder, generator, discriminator
from sp_ops import L1_loss

from libs.mb_melgan.configs.mb_melgan import MultiBandMelGANGeneratorConfig
from libs.mb_melgan.models.mb_melgan import TFPQMF, TFMelGANGenerator
from utils import mag_to_mel

import tensorflow as tf
import numpy as np
import os.path as op
import os
import datetime
import math
import yaml

print(tf.__version__)


class hp:
    # Training setting
    data_file = 'ljs'   # esc50, maestro, ljs
    isMag = True
    labeled = False  # 15:True, large:False
    save_descript = '_spadeNet_bs1_AllWeighted_Novocol_m100'
    debug_graph = False
    # training_file = op.join('./data', data_file, 'train_list.txt')
    testing_file = op.join('./data', data_file, 'test_list.txt')
    logdir = op.join('./test_logs', f'{data_file}{save_descript}')
    is_output = True
    is_output_wav = False
    output_dir = op.join(logdir, 'output')
    wave_dir = op.join(output_dir, 'wave')
    wav_ext = '.wav'
    # pretrain_model = 'pretrain_models/first_stage'
    checkpoint_prefix = op.join(logdir, "ckpt")
    checkpoint_restore_dir = './logs/ljs_spadeNet_bs16_AllWeighted_Novocol_m10_110'
    checkpoint_freq = 10000
    restore_epochs = 0  # Specify for restore training.
    epochs = 1
    # summary_freq = 50
    steps_per_epoch = -1  # -1: whole training data.
    batch_size = 1
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
    sr = 22050          # ljs: 22050, others: 44100
    hop_size = 256
    image_height = 256
    image_width = 256
    image_channel = 1
    length_5sec = int((sr / hop_size) * 5)              # int() = floor()
    # mask_height = 256
    # mask_width = round(length_5sec * 0.2 * 1.1)        # max of mask width
    # max_delta_height = 0
    # max_delta_width = round(length_5sec * 0.2 * 0.2)    # decrease with this delta
    # vertical_margin = 0
    # horizontal_margin = 0
    seg_middle = round(length_5sec * 0.2 * 3.5)        # esc50: 0, others: 2.75
    seg_start = seg_middle - (image_width // 2)
    # seg_start = 0
    mask_pth = '/work/r08922a13/datasets/mask/mask_time_086.npy'
    # 034 068 104 138 172
    # 018 034 052 068 086
    ir_mask = False
    # Vocoder
    v_ckpt = f'libs/mb_melgan/ckpt/{data_file}/generator-800000.h5'
    v_config = f'libs/mb_melgan/configs/multiband_melgan.{data_file}_v1.yaml'


if __name__ == "__main__":

    # make output dir
    if op.isdir(hp.logdir) is False:
        os.mkdir(hp.logdir)
        os.mkdir(hp.output_dir)
        os.mkdir(hp.wave_dir)

    # load mask
    mask = np.load(hp.mask_pth)
    mask = mask / 256
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)

    # load data
    print("Load data from path...")
    test_data_fnames, test_labels = load_data_filename(hp.testing_file, hp.labeled)
    print(test_data_fnames.shape)
    print(test_data_fnames[0])

    # Initialize distribute strategy
    # strategy = tf.distribute.MirroredStrategy()

    # It seems that the auto share is not avalibel for our data type.
    # How ever it should be tf.data.experimental.AutoShardPolicy.FILE since the files >> workers.
    # If transfering data type to TFRecord, it will be good to FILE share policy.
    # options = tf.data.Options()
    # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    # GLOBAL_BATCH_SIZE = hp.batch_size * strategy.num_replicas_in_sync

    # padded_shape = [hp.image_height, None]

    print("Prefetching...")
    # Map function should update to TFRecord
    # instead of tf.py_function for better performance.
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data_fnames))
    test_dataset = test_dataset.map(lambda x: tf.numpy_function(load_npy_test, inp=[x, hp.length_5sec], Tout=[tf.float32, tf.string]),
                                    num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(hp.batch_size, drop_remainder=True)
    # test_dataset = test_dataset.with_options(options)

    # print(f'training list\'s shape:{train_data.shape}, testing list\'s shape: {test_data.shape}')

    # to distributed strategy
    # test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

    # Create Tensorboard Writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = op.join(hp.logdir, current_time)
    file_summary_writer = tf.summary.create_file_writer(log_dir)

    # whole training data
    if hp.steps_per_epoch == -1:
        test_steps_per_epoch = math.ceil(test_data_fnames.shape[0] / hp.batch_size)
    elif hp.steps_per_epoch > 0:
        test_steps_per_epoch = int(hp.steps_per_epoch)
    else:
        raise ValueError(f"Wrong number assigned to steps_per_epoch: {hp.steps_per_epoch}")
    # Skip the last batches for the batch size consistency, due to error might occur in CA module.
    test_steps_per_epoch -= 1

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
        decay_steps=test_steps_per_epoch*1000,
        decay_rate=1,
        staircase=False)
    d_lr = tf.keras.optimizers.schedules.InverseTimeDecay(
        2e-4,
        decay_steps=test_steps_per_epoch*1000,
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
                    #first_stage=first_stage,
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
    test_accuracy = tf.keras.metrics.MeanAbsoluteError(name='test_MAE_loss')

    def test_step(inputs):
        loss = {}

        x_ori, mask = inputs
        rands = hp.seg_start
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
        # complete = tf.concat([pre, x_complete, post], axis=2)
        complete = tf.concat([pre, x_stage2, post], axis=2)
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

        g_2st_loss = hp.l1_alpha * L1_loss(x_pos, x_complete, True)
        g_2st_loss_noW = hp.l1_alpha * L1_loss(x_pos, x_complete, False)

        # t_loss = g_2st_loss
        # t_loss = loss_fn(x_pos, x_stage2)

        loss['test/mae'] = g_2st_loss_noW
        loss['test/weighted mae'] = g_2st_loss
        loss['test/psnr'] = tf.math.reduce_mean(tf.image.psnr(x_pos, x_complete, max_val=1.0))
        loss['test/ssim'] = tf.math.reduce_mean(tf.image.ssim(x_pos, x_complete, max_val=1.0))
        loss['test/BG_loss'] = L1_loss(x_stage2, x_complete, False)

        test_loss.update_state(g_2st_loss_noW)
        test_weighted_loss.update_state(g_2st_loss)
        test_accuracy.update_state(x_pos, x_complete)

        #summary_images = [x_incomplete, x_stage1, x_stage2, x_complete, x_pos, semap]
        summary_images = [x_incomplete, x_stage2, x_complete, x_pos]
        summary_images = tf.concat(summary_images, axis=2)

        summary_audios = tf.concat([x_audios, pos_audios, incomplete_audios], axis=2)

        return loss, summary_images, summary_audios

        # @tf.function
        # def distributed_test_step(dataset_inputs):
        #     summary_images, summary_audios = strategy.run(test_step, args=(dataset_inputs,))
        #     return summary_images, summary_audios

        # Experiment Epoch
    for epoch in range(hp.restore_epochs, hp.epochs+hp.restore_epochs):
        total_loss = {}
        num_batches = 0

        ### Testing loop
        test_iter = iter(test_dataset)
        for batch_step in tqdm(range(test_steps_per_epoch)):
            step = epoch * test_steps_per_epoch + batch_step
            # mask = create_mask()
            x_ori, filename = next(test_iter)
            step_loss, test_images, test_audios = test_step([x_ori, mask])

            for key in step_loss:
                if key in total_loss:
                    total_loss[key] += step_loss[key]
                else:
                    total_loss[key] = step_loss[key]
            num_batches += 1

            img_incomplete, img_stage2, img_complete, img_pos = tf.split(test_images, 4, axis=2)
            test_x, test_pos, test_incomplete = tf.split(test_audios, 3, axis=2)
            with file_summary_writer.as_default():
                dict_scalar_summary('test loss step', step_loss, step=step)
                images_summary("Testing result/images", test_images, step=step, max_outputs=hp.max_outputs)
                audio_summary("Testing result/x_complete", test_x, hp.sr, step=step, max_outputs=hp.max_outputs)
                audio_summary("Testing result/x_pos", test_pos, hp.sr, step=step, max_outputs=hp.max_outputs)
                audio_summary("Testing result/x_incomplete", test_incomplete, hp.sr, step=step, max_outputs=hp.max_outputs)

            if hp.is_output:
                assert hp.batch_size == 1, print("Batch_size should be 1 for save output.")

                incomplete_name = op.basename(str(filename[0])).split('-mag-')[0] + '_incomplete'
                np_incomplete_name = op.join(hp.output_dir, incomplete_name)
                wav_incomplete_name = op.join(hp.wave_dir, incomplete_name) + hp.wav_ext
                np.save(np_incomplete_name, img_incomplete[0])
                if hp.is_output_wav:
                    incomplete_wav = tf.audio.encode_wav(test_incomplete[0], hp.sr)
                    tf.io.write_file(wav_incomplete_name, incomplete_wav)

                complete_name = op.basename(str(filename[0])).split('-mag-')[0] + '_complete'
                np_complete_name = op.join(hp.output_dir, complete_name)
                wav_complete_name = op.join(hp.wave_dir, complete_name) + hp.wav_ext
                np.save(np_complete_name, img_complete[0])
                if hp.is_output_wav:
                    complete_wav = tf.audio.encode_wav(test_x[0], hp.sr)
                    tf.io.write_file(wav_complete_name, complete_wav)

                stage2_name = op.basename(str(filename[0])).split('-mag-')[0] + '_stage2'
                np_stage2_name = op.join(hp.output_dir, stage2_name)
                np.save(np_stage2_name, img_stage2[0])

                ori_name = op.basename(str(filename[0])).split('-mag-')[0] + '_ori'
                np_ori_name = op.join(hp.output_dir, ori_name)
                wav_ori_name = op.join(hp.wave_dir, ori_name) + hp.wav_ext
                np.save(np_ori_name, img_pos[0])
                if hp.is_output_wav:
                    ori_wav = tf.audio.encode_wav(test_pos[0], hp.sr)
                    tf.io.write_file(wav_ori_name, ori_wav)

        for key in total_loss:
            total_loss[key] = total_loss[key] / num_batches
            print(f'Loss: {key} is value: {total_loss[key]}')

            # Write to tensorboard.
        with file_summary_writer.as_default():
            dict_scalar_summary('test loss epoch', total_loss, step=epoch)
            scalar_summary('test loss', test_loss.result(), step=epoch)
            scalar_summary('test loss/weighted', test_weighted_loss.result(), step=epoch)
            scalar_summary('test MAE loss', test_accuracy.result(), step=epoch)

        test_loss.reset_states()
        test_weighted_loss.reset_states()
        test_accuracy.reset_states()
