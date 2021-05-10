from inpaint_ops import random_bbox, bbox2mask
from inpaint_ops import brush_stroke_mask
from inpaint_ops import mag_mel_weighted_map
from tqdm import tqdm
from SPADE import image_encoder, generator, discriminator

from pretrain.models import coarse_inpaint_net

from libs.mb_melgan.configs.mb_melgan import MultiBandMelGANGeneratorConfig
from libs.mb_melgan.models.mb_melgan import TFPQMF, TFMelGANGenerator
from utils import mag_to_mel, toSpec_db_norm

import tensorflow as tf
import os.path as op
import os
import yaml
import argparse


print(tf.__version__)


parser = argparse.ArgumentParser()
parser.add_argument('--test_audio', default='', type=str,
                    help='The filename of test data.')


class hp:
    # Training setting
    data_file = 'esc50_mag'
    isMag = True
    labeled = False  # 15:True, large:False
    checkpoint_restore_dir = '/work/r08922a13/audio_inpainting/logs/esc50_mag_spadeNet_GPU8_Voc_noF1'
    #testing_file = op.join('./data', data_file, 'test_list.txt')
    #logdir = op.join('./logs', f'{data_file}{save_descript}')
    output_dir = op.join('./output', op.basename(checkpoint_restore_dir))
    wav_ext = '.wav'
    img_ext = '.png'
    #checkpoint_prefix = op.join(logdir, "ckpt")
    #checkpoint_freq = 100
    #restore_epochs = 0  # Specify for restore training.
    #epochs = 10000
    #summary_freq = 50
    steps_per_epoch = -1  # -1: whole training data.
    batch_size = 16
    #max_outputs = 5
    #profile = False  # profile on first epoch, batch 10~20.
    l1_alpha = 1.
    weighted_loss = False
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
    horizontal_margin = 0
    ir_mask = False
    # Vocoder
    sr = 44100
    v_ckpt = 'libs/mb_melgan/ckpt/generator-800000.h5'
    v_config = 'libs/mb_melgan/configs/multiband_melgan.v1.yaml'


if __name__ == "__main__":
    args = parser.parse_args()
    test_audio = args.test_audio

    if op.isfile(test_audio) is False:
        raise ValueError("The data is not exist with path:", test_audio)
    if op.isdir(op.dirname(hp.output_dir)) is False:
        os.mkdir(op.dirname(hp.output_dir))
    if op.isdir(hp.output_dir) is False:
        os.mkdir(hp.output_dir)

    raw_audio = tf.io.read_file(test_audio)
    waveform, sr = tf.audio.decode_wav(raw_audio)
    print(f"Read the test data with shape: {waveform.shape}, sr: {sr}.")
    if sr != hp.sr:
        raise ValueError("Unexpected sample rate:", sr)

    if waveform.shape[-1] == 1:
        is_stereo = False
    elif waveform.shape[-1] == 2:
        is_stereo = True
    else:
        raise ValueError(f"Unexpected channel of test_audio, should be 1 or 2: {waveform.shape}")

    test_epoch = 0
    last_spec = None

    if is_stereo:
        waveform_track1, waveform_track2 = tf.split(waveform, 2, axis=1)
    else:
        waveform = tf.reshape(waveform, waveform.shape[0])
        spec = toSpec_db_norm(waveform)
        # mag shape: [T, n_fft//2] -> [n_fft//2, T]
        spec = tf.transpose(spec, perm=[1, 0])

        if spec.shape[1] < hp.image_width:
            raise ValueError(f"The length of test audio is too short: {waveform.shape}, sr:{sr}.")
        else:
            if spec.shape[1] == hp.image_width:
                raise NotImplementedError("Not impletment the part of just the same size.")
                test_batch = 1
                test_epoch = 1
                test_dataset = spec
            else:
                specs = []
                if spec.shape[1] % hp.image_width != 0:
                    batch = int(tf.math.floor(spec.shape[1] / hp.image_width))
                    for part in range(batch):
                        per_spec = spec[:, part*hp.image_width: (part+1)*hp.image_width]
                        per_spec = tf.reshape(per_spec, per_spec.shape[:] + [1])
                        specs.append(per_spec)
                    last_spec = spec[:, batch*hp.image_width:]
                    #last_spec = tf.reshape(last_spec, [last_spec.shape[:], 1])

                if batch <= hp.batch_size:
                    test_batch = batch
                    test_epoch = 1
                else:
                    test_batch = hp.batch_size
                    test_epoch = tf.math.ceil(batch / hp.batch_size)

                test_dataset = tf.data.Dataset.from_tensor_slices((specs))
                test_dataset = test_dataset.batch(test_batch)

    print("Build model...")
    # build first model, load pretrain weight, freeze weights.
    #first_stage, optimizer = coarse_inpaint_net(hp.image_height, hp.image_width, hp.image_channel)
    # first_ckpt = tf.train.Checkpoint(
    #     optimizer=optimizer,
    #     model=first_stage)
    # first_ckpt.restore(tf.train.latest_checkpoint(hp.pretrain_weights_dir))
    # first_stage = tf.keras.models.load_model(hp.pretrain_model)
    #print(first_stage.summary())
    #for layer in first_stage.layers:
    #    layer.trainable = False

    # subbands = mb_melgan(mel), audios = pqmf.synthesis(subbands)
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
        decay_steps=0,
        decay_rate=1,
        staircase=False)
    d_lr = tf.keras.optimizers.schedules.InverseTimeDecay(
        2e-4,
        decay_steps=0,
        decay_rate=1,
        staircase=False)
    beta_1 = 0.0
    beta_2 = 0.9

    encoder = image_encoder(hp.image_height, hp.image_width, hp.image_channel)
    generator = generator(hp.image_height, hp.image_width, hp.image_channel, test_batch)
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

    # Would combine with loss_fn
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.MeanAbsoluteError(name='test_MAE_loss')

    # The function here only consider the first dimension(B) of losses,
    # wont mean over dims: H, W, C, due to reduce_sup in methods.
    # Calculate the distributed losses by myself.
    def compute_global_loss(loss):
        return loss

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

    def test_step(inputs):
        x_pos, mask = inputs

        x_incomplete = x_pos * (1.-mask)
        '''
        first_stage_input = [x_incomplete, mask]
        x_stage1 = first_stage(inputs=first_stage_input, training=False)
        semap = x_incomplete + x_stage1 * mask
        '''
        semap = x_incomplete
        x_mean, x_var = encoder(inputs=semap, training=False)
        G_input = [semap, x_mean, x_var]
        x_stage2 = generator(inputs=G_input, training=False)
        x_complete = x_incomplete + x_stage2 * mask

        g_2st_diff = tf.math.abs(x_pos - x_stage2)
        if hp.weighted_loss:
            g_2st_diff = mag_mel_weighted_map(g_2st_diff)
        g_2st_loss = hp.l1_alpha * tf.math.reduce_mean(g_2st_diff)

        t_loss = g_2st_loss
        test_loss.update_state(t_loss)
        test_accuracy.update_state(x_pos, x_complete)

        summary_images = [x_incomplete, x_complete, x_pos]
        summary_images = tf.concat(summary_images, axis=2)

        #summary_audios = tf.concat([x_audios, pos_audios, incomplete_audios], axis=2)

        return summary_images

    x_incomplete = []
    x_complete = []
    x_origin = []
    num_pad_data = 0

    ### Testing loop
    test_iter = iter(test_dataset)
    for batch_step in tqdm(range(test_epoch)):
        mask = create_mask()
        x_pos = next(test_iter)
        if x_pos.shape[0] < test_batch:
            num_pad_data = test_batch - x_pos.shape[0]
            pad_data = tf.zeros_like(x_pos[0])
            for i in range(num_pad_data-1):
                pad_data = tf.concat([pad_data, tf.zeros_like(x_pos[0])], axis=0)

            x_pos = tf.concat([x_pos, pad_data], axis=0)

        test_images = test_step([x_pos, mask])

        #test_x, test_pos, test_incomplete = tf.split(test_audios, 3, axis=2)
        incomplete, complete, pos = tf.split(test_images, 3, axis=2)
        if num_pad_data > 0:
            incomplete = incomplete[:-num_pad_data]
            complete = complete[:-num_pad_data]
            pos = pos[:-num_pad_data]

        incomplete = tf.split(incomplete, incomplete.shape[0], axis=0)
        incomplete = tf.concat(incomplete, axis=2)
        incomplete = tf.reshape(incomplete, incomplete.shape[1:-1])

        complete = tf.split(complete, complete.shape[0], axis=0)
        complete = tf.concat(complete, axis=2)
        complete = tf.reshape(complete, complete.shape[1:-1])

        pos = tf.split(pos, pos.shape[0], axis=0)
        pos = tf.concat(pos, axis=2)
        pos = tf.reshape(pos, pos.shape[1:-1])

        x_incomplete.append(incomplete)
        x_complete.append(complete)
        x_origin.append(pos)

    if last_spec is not None:
        x_incomplete.append(last_spec)
        x_complete.append(last_spec)
        x_origin.append(last_spec)

    x_incomplete = tf.concat(x_incomplete, axis=1)
    x_complete = tf.concat(x_complete, axis=1)
    x_origin = tf.concat(x_origin, axis=1)

    if hp.isMag:
        pos_mels = mag_to_mel(x_origin)
        incomplete_mels = mag_to_mel(x_incomplete)
        x_mels = mag_to_mel(x_complete)
    else:
        pos_mels = x_origin
        incomplete_mels = x_incomplete
        x_mels = x_complete

    basename = op.basename(test_audio).split('.')[0]

    x_subbands = mb_melgan(x_mels)
    x_audios = pqmf.synthesis(x_subbands)
    x_audios = tf.reshape(x_audios, x_audios.shape[1:])
    audio_name = f'{basename}_inpainted{hp.wav_ext}'
    path = op.join(hp.output_dir, audio_name)
    raw_wav = tf.audio.encode_wav(x_audios, sr)
    tf.io.write_file(path, raw_wav)

    img_name = f'{basename}_inpainted{hp.img_ext}'
    path = op.join(hp.output_dir, img_name)
    x_complete = tf.reshape(x_complete, x_complete.shape + [1])
    tf.keras.preprocessing.image.save_img(path, x_complete)

    pos_subbands = mb_melgan(pos_mels)
    pos_audios = pqmf.synthesis(pos_subbands)
    pos_audios = tf.reshape(pos_audios, pos_audios.shape[1:])
    audio_name = f'{basename}_origin{hp.wav_ext}'
    path = op.join(hp.output_dir, audio_name)
    raw_wav = tf.audio.encode_wav(x_audios, sr)
    tf.io.write_file(path, raw_wav)

    img_name = f'{basename}_origin{hp.img_ext}'
    path = op.join(hp.output_dir, img_name)
    x_origin = tf.reshape(x_origin, x_origin.shape + [1])
    tf.keras.preprocessing.image.save_img(path, x_origin)

    incomplete_subbands = mb_melgan(incomplete_mels)
    incomplete_audios = pqmf.synthesis(incomplete_subbands)
    incomplete_audios = tf.reshape(incomplete_audios, incomplete_audios.shape[1:])
    audio_name = f'{basename}_masked{hp.wav_ext}'
    path = op.join(hp.output_dir, audio_name)
    raw_wav = tf.audio.encode_wav(x_audios, sr)
    tf.io.write_file(path, raw_wav)

    img_name = f'{basename}_masked{hp.img_ext}'
    path = op.join(hp.output_dir, img_name)
    x_incomplete = tf.reshape(x_incomplete, x_incomplete.shape + [1])
    tf.keras.preprocessing.image.save_img(path, x_incomplete)

    template = ("Test Loss: {}, Test MAE loss: {}")
    print(template.format(test_loss.result(),
                          test_accuracy.result()*100))
    test_loss.reset_states()
    test_accuracy.reset_states()
