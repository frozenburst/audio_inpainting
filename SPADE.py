import tensorflow as tf

from tensorflow import keras
import tensorflow_addons as tfa

from sp_ops import sp_conv, fully_connected
from sp_ops import spade_resblock
from sp_ops import z_sample, up_sample, down_sample_avg


# Reference from NVIDIA SPADE, taki0112/SPADE-Tensorflow Github
def image_encoder(img_height=256, img_width=256, img_channel=1, training=True):
    xin = keras.Input(shape=(img_height, img_width, img_channel), name="image")

    cnum = 64
    alpha = 0.2
    sn = True

    channel = cnum
    x = sp_conv(xin, channel, kernel=3, stride=2, pad=1, use_bias=True, sn=sn, name='spconv')
    x = tfa.layers.InstanceNormalization(epsilon=1e-05, center=True, scale=True, name='ins_norm')(x)

    for i in range(3):
        # 128, 256, 512
        channel = channel * 2
        x = tf.keras.layers.LeakyReLU(alpha)(x)
        x = sp_conv(x, channel, kernel=3, stride=2, pad=1, use_bias=True, sn=sn, name='spconv_' + str(i))
        x = tfa.layers.InstanceNormalization(epsilon=1e-05, center=True, scale=True, name='ins_norm' + str(i))(x)

    x = tf.keras.layers.LeakyReLU(alpha)(x)
    x = sp_conv(x, channel, kernel=3, stride=2, pad=1, use_bias=True, sn=sn, name='spconv_3')
    x = tfa.layers.InstanceNormalization(epsilon=1e-05, center=True, scale=True, name='ins_norm_3')(x)

    if img_height >= 256 or img_width >= 256:
        x = tf.keras.layers.LeakyReLU(alpha)(x)
        x = sp_conv(x, channel, kernel=3, stride=2, pad=1, use_bias=True, sn=sn, name='spconv_4')
        x = tfa.layers.InstanceNormalization(epsilon=1e-05, center=True, scale=True, name='ins_norm_4')(x)

    x = tf.keras.layers.LeakyReLU(alpha)(x)

    mean = fully_connected(x, channel // 2, use_bias=True, sn=sn, name='linear_mean')
    var = fully_connected(x, channel // 2, use_bias=True, sn=sn, name='linear_var')

    image_encoder = keras.Model(inputs=xin, outputs=[mean, var], name='image_encoder')
    return image_encoder


def generator(img_height=256, img_width=256, img_channel=1, batch_size=16, linear_length=256, training=True, name="G"):
    semap = keras.Input(shape=(img_height, img_width, img_channel), name="semap")
    x_mean = keras.Input(shape=(linear_length), name="mean")
    x_var = keras.Input(shape=(linear_length), name="var")

    cnum = 64 * 4 * 4
    sn = True
    alpha = 0.2

    x = z_sample(x_mean, x_var)

    num_up_layers = 6
    """
    # If num_up_layers = 5 (normal)
    # 64x64 -> 2
    # 128x128 -> 4
    # 256x256 -> 8
    # 512x512 -> 16
    """
    z_width = img_width // (pow(2, num_up_layers))
    z_height = img_height // (pow(2, num_up_layers))
    # 4 * 4
    x = fully_connected(x, units=z_height * z_width * cnum, use_bias=True, sn=False, name=f'{name}_linear_x')
    x = tf.reshape(x, [batch_size, z_height, z_width, cnum])

    x = spade_resblock(semap, x, channels=cnum, use_bias=True, sn=sn, name=f'{name}_spade_resblock_fix_0')
    # 8 * 8
    x = up_sample(x, scale_factor=2)
    x = spade_resblock(semap, x, channels=cnum, use_bias=True, sn=sn, name=f'{name}_spade_resblock_fix_1')
    # 16 * 16
    x = up_sample(x, scale_factor=2)
    x = spade_resblock(semap, x, channels=cnum, use_bias=True, sn=sn, name=f'{name}_spade_resblock_fix_2')

    for i in range(4):
        cnum = cnum // 2
        x = up_sample(x, scale_factor=2)
        x = spade_resblock(semap, x, channels=cnum, use_bias=True, sn=sn, name=f'{name}_spade_resblock_{i}')
        # (H, W) 32 64 128 256

    x = tf.keras.layers.LeakyReLU(alpha)(x)
    x = sp_conv(x, channels=img_channel, kernel=3, stride=1, pad=1, use_bias=True, sn=False, name=f'{name}_logit')
    x = tf.keras.activations.tanh(x)

    outputs = x

    generator = keras.Model(inputs=[semap, x_mean, x_var], outputs=outputs, name='G')
    return generator


def discriminator(img_height=256, img_width=256, img_channel=1, training=True, name="D"):
    semap = keras.Input(shape=(img_height, img_width, img_channel), name='semap')
    x_init = keras.Input(shape=(img_height, img_width, img_channel), name='x_init')

    inputs = [semap, x_init]
    cnum = 64
    alpha = 0.2
    D_logit = []
    n_scale = 2
    n_dis = 4
    sn = True
    for scale in range(n_scale):
        feature_loss = []
        channel = cnum
        x = tf.concat([semap, x_init], axis=-1)

        x = sp_conv(x, channel, kernel=4, stride=2, pad=1, use_bias=True, sn=False, name=f'ms_{scale}_conv_0')
        x = tf.keras.layers.LeakyReLU(alpha)(x)

        feature_loss.append(x)

        for i in range(1, n_dis):
            stride = 1 if i == n_dis - 1 else 2
            channel = min(channel * 2, 512)

            x = sp_conv(x, channel, kernel=4, stride=stride, pad=1, use_bias=True, sn=sn, name=f'ms_{scale}_conv_{i}')
            x = tfa.layers.InstanceNormalization(epsilon=1e-05, center=True, scale=True, name=f'ms_{scale}_ins_norm_{i}')(x)
            x = tf.keras.layers.LeakyReLU(alpha)(x)

            feature_loss.append(x)

        x = sp_conv(x, channels=1, kernel=4, stride=1, pad=1, use_bias=True, sn=sn, name=f'ms_{scale}_D_logit')

        feature_loss.append(x)
        D_logit.append(feature_loss)

        x_init = down_sample_avg(x_init, name=f'ms_{scale}_xin_ds_avg')
        semap = down_sample_avg(semap, name=f'ms_{scale}_semap_ds_avg')

    outputs = D_logit

    discriminator = keras.Model(inputs=inputs, outputs=outputs, name='D')
    return discriminator
