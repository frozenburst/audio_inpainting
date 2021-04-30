import tensorflow as tf

from tensorflow.keras import layers
from utils import pytorch_xavier_weight_factor, pytorch_kaiming_weight_factor
from sn_layers import SNConv2D, SNDense


# Reference from GitHub: https://github.com/taki0112/SPADE-Tensorflow
factor, mode, uniform = pytorch_xavier_weight_factor(gain=0.02, uniform=False)
# This is different from original with normal stddev factor 1.3^2 -> 1.13684..
weight_init = tf.keras.initializers.VarianceScaling(scale=factor, mode=mode, distribution='truncated_normal')


weight_regularizer = None
weight_regularizer_fully = None


def sp_conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, name='sp_conv0'):
    if pad > 0:
        h = x.get_shape().as_list()[1]
        if h % stride == 0:
            pad = pad * 2
        else:
            pad = max(kernel - (h % stride), 0)

        pad_top = pad // 2
        pad_bottom = pad - pad_top
        pad_left = pad // 2
        pad_right = pad - pad_left

        if pad_type == 'zero':
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
        elif pad_type == 'reflect':
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

    if sn:
        x = SNConv2D(channels, kernel, (stride, stride), padding='valid', activation=None,
                     kernel_regularizer=weight_regularizer, kernel_initializer=weight_init, name=name)(x)
    else:
        x = layers.Conv2D(channels, kernel, (stride, stride), padding='valid', activation=None,
                          kernel_regularizer=weight_regularizer, kernel_initializer=weight_init, name=name)(x)

    return x


def fully_connected(x, units, use_bias=True, sn=False, name='linear'):
    x = layers.Flatten()(x)
    if sn:
        x = SNDense(units=units, kernel_initializer=weight_init,
                    kernel_regularizer=weight_regularizer_fully,
                    use_bias=use_bias, name=name)(x)
    else:
        x = layers.Dense(units=units, kernel_initializer=weight_init,
                         kernel_regularizer=weight_regularizer_fully,
                         use_bias=use_bias, name=name)(x)
    return x


##################################################################################
# Residual-block
##################################################################################

def spade_resblock(segmap, x_init, channels, use_bias=True, sn=False, name='spade_resblock'):
    channel_in = x_init.get_shape().as_list()[-1]
    channel_middle = min(channel_in, channels)
    # pytorch alpha is 0.01
    alpha = 0.2

    x = spade(segmap, x_init, channel_in, use_bias=use_bias, sn=False, name=name+'_spade_1')
    x = tf.keras.layers.LeakyReLU(alpha)(x)
    x = sp_conv(x, channels=channel_middle, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn, name=name+'_conv1')

    x = spade(segmap, x, channels=channel_middle, use_bias=use_bias, sn=False, name=name+'_spade_2')
    x = tf.keras.layers.LeakyReLU(alpha)(x)
    x = sp_conv(x, channels=channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn, name=name+'_conv2')

    if channel_in != channels:
        x_init = spade(segmap, x_init, channels=channel_in, use_bias=use_bias, sn=False, name=name+'_spade_shortcut')
        x_init = sp_conv(x_init, channels=channels, kernel=1, stride=1, use_bias=False, sn=sn, name=name+'_conv_shortcut')

    return x + x_init


def spade(segmap, x_init, channels, use_bias=True, sn=False, name='spade'):
    x = param_free_norm(x_init)

    _, x_h, x_w, _ = x_init.get_shape().as_list()
    _, segmap_h, segmap_w, _ = segmap.get_shape().as_list()

    factor_h = segmap_h // x_h  # 256 // 4 = 64
    factor_w = segmap_w // x_w

    segmap_down = down_sample(segmap, factor_h, factor_w)

    segmap_down = sp_conv(segmap_down, channels=128, kernel=5, stride=1, pad=2, use_bias=use_bias, sn=sn, name=name+'_conv128')
    segmap_down = tf.keras.layers.ReLU()(segmap_down)

    segmap_gamma = sp_conv(segmap_down, channels=channels, kernel=5, stride=1, pad=2, use_bias=use_bias, sn=sn, name=name+'_conv_gamma')
    segmap_beta = sp_conv(segmap_down, channels=channels, kernel=5, stride=1, pad=2, use_bias=use_bias, sn=sn, name=name+'_conv_beta')

    x = x * (1 + segmap_gamma) + segmap_beta
    return x


def param_free_norm(x, epsilon=1e-5):
    x_mean, x_var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    x_std = tf.math.sqrt(x_var + epsilon)
    return (x - x_mean) / x_std


##################################################################################
# Sampling
##################################################################################

def down_sample(x, scale_factor_h, scale_factor_w):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h // scale_factor_h, w // scale_factor_w]
    return tf.image.resize(x, size=new_size, method='bilinear')


def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize(x, size=new_size, method='bilinear')


def down_sample_avg(x, scale_factor=2, name='ds_avg'):
    return tf.keras.layers.AveragePooling2D(pool_size=3, strides=scale_factor, padding="same", name=name)(x)


def z_sample(mean, logvar):
    eps = tf.random.normal(tf.shape(mean), mean=0.0, stddev=1.0, dtype=tf.dtypes.float32)
    return mean + tf.math.exp(logvar * 0.5) * eps


##################################################################################
# Loss function
##################################################################################


def L1_loss(x, y):
    loss = tf.math.reduce_mean(tf.math.abs(x - y))
    return loss


def generator_loss(fake):
    loss = []
    fake_loss = 0

    for i in range(len(fake)):
        # -1 means the last feature layer.
        fake_loss = -tf.math.reduce_mean(fake[i][-1])
        loss.append(fake_loss)

    return tf.math.reduce_mean(loss)


def discriminator_loss(real, fake):
    loss = []
    real_loss = 0
    fake_loss = 0

    for i in range(len(fake)):
        # -1 means the last feature, only feature loss need all pairs.
        # here is opposite way to cal hinge loss.
        real_loss = -tf.math.reduce_mean(tf.math.minimum(real[i][-1] - 1, 0.0))
        fake_loss = -tf.math.reduce_mean(tf.math.minimum(-fake[i][-1] - 1, 0.0))

        loss.append(real_loss + fake_loss)

    return tf.math.reduce_mean(loss)


def feature_loss(real, fake):
    loss = []

    for i in range(len(fake)):
        intermediate_loss = 0
        for j in range(len(fake[i]) - 1):
            intermediate_loss += L1_loss(real[i][j], fake[i][j])
        loss.append(intermediate_loss)

    return tf.math.reduce_mean(loss)


def kl_loss(mean, logvar):
    # shape : [batch_size, channel]
    loss = 0.5 * tf.reduce_sum(tf.square(mean) + tf.exp(logvar) - 1 - logvar)
    return loss
