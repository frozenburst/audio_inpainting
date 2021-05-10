from data_loader import load_npy, load_data_filename
from summary_ops import scalar_summary, images_summary
from summary_ops import dict_scalar_summary
from summary_ops import gradient_calc
from summary_ops import audio_summary
from inpaint_ops import random_bbox, bbox2mask, gan_hinge_loss
from inpaint_ops import brush_stroke_mask
from inpaint_ops import mag_mel_weighted_map
from tqdm import tqdm
from SPADE import image_encoder, generator, discriminator
from sp_ops import generator_loss, discriminator_loss
from sp_ops import feature_loss, kl_loss, L1_loss

from pretrain.models import coarse_inpaint_net

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
    data_file = 'esc50_mag'
    isMag = True
    labeled = False  # 15:True, large:False
    checkpoint_restore_dir = ''
    testing_file = op.join('./data', data_file, 'test_list.txt')
    #logdir = op.join('./logs', f'{data_file}{save_descript}')
    output_dir = op.join('./output', op.basename(checkpoint_restore_dir))
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
    image_width = 80
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

    # load data
    print("Load data from path...")
    test_data_fnames, test_labels = load_data_filename(hp.testing_file, hp.labeled)
    print(test_data_fnames.shape)

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
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data_fnames))
    test_dataset = test_dataset.map(lambda x: tf.numpy_function(load_npy, inp=[x], Tout=tf.float32),
                                    num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(GLOBAL_BATCH_SIZE)
    test_dataset = test_dataset.with_options(options)

    # print(f'training list\'s shape:{train_data.shape}, testing list\'s shape: {test_data.shape}')

    # to distributed strategy
    test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

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

        # Would combine with loss_fn
        test_loss = tf.keras.metrics.Mean(name='test_loss')
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

            if hp.isMag:
                pos_mels = mag_to_mel(x_pos)
                incomplete_mels = mag_to_mel(x_incomplete)
                x_mels = mag_to_mel(x_complete)
            else:
                pos_mels = x_pos
                incomplete_mels = x_incomplete
                x_mels = x_complete

            x_subbands = mb_melgan(x_mels)
            x_audios = pqmf.synthesis(x_subbands)

            pos_subbands = mb_melgan(pos_mels)
            pos_audios = pqmf.synthesis(pos_subbands)

            incomplete_subbands = mb_melgan(incomplete_mels)
            incomplete_audios = pqmf.synthesis(incomplete_subbands)

            g_2st_diff = tf.math.abs(x_pos - x_stage2)
            if hp.weighted_loss:
                g_2st_diff = mag_mel_weighted_map(g_2st_diff)
            g_2st_loss = hp.l1_alpha * tf.math.reduce_mean(g_2st_diff)

            t_loss = g_2st_loss
            test_loss.update_state(t_loss)
            test_accuracy.update_state(x_pos, x_complete)

            summary_images = [x_incomplete, x_stage2, x_complete, x_pos]
            summary_images = tf.concat(summary_images, axis=2)

            summary_audios = tf.concat([x_audios, pos_audios, incomplete_audios], axis=2)

            return summary_images, summary_audios

        @tf.function
        def distributed_test_step(dataset_inputs):
            summary_images, summary_audios = strategy.run(test_step, args=(dataset_inputs,))
            return summary_images, summary_audios

        ### Testing loop
        test_iter = iter(test_dist_dataset)
        for batch_step in tqdm(range(test_steps_per_epoch)):
            mask = create_mask()
            x_pos = next(test_iter)
            test_images, test_audios = distributed_test_step([x_pos, mask])

            if strategy.num_replicas_in_sync > 1:
                test_images = test_images.values
                test_images = tf.concat(test_images, axis=0)

                test_audios = test_audios.values
                test_audios = tf.concat(test_audios, axis=0)
            test_x, test_pos, test_incomplete = tf.split(test_audios, 3, axis=2)


            images_summary("Testing result/images", test_images, step=step, max_outputs=hp.max_outputs)
            audio_summary("Testing result/x_complete", test_x, hp.sr, step=step, max_outputs=hp.max_outputs)
            audio_summary("Testing result/x_pos", test_pos, hp.sr, step=step, max_outputs=hp.max_outputs)
            audio_summary("Testing result/x_incomplete", test_incomplete, hp.sr, step=step, max_outputs=hp.max_outputs)

        template = ("Test Loss: {}, Test MAE loss: {}")
        print(template.format(test_loss.result(),
                                test_accuracy.result()*100))
        test_loss.reset_states()
        test_accuracy.reset_states()
