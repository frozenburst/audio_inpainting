import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from inpaint_ops import gen_conv
from sn_conv import SNConv2D


# Reference from GitHub: https://github.com/JiahuiYu/generative_inpainting
def coarse_inpaint_net(img_height=256, img_width=256, img_c=1, training=True):

    xin = keras.Input(shape=(img_height, img_width, img_c), name="image")
    mask = keras.Input(shape=(img_height, img_width, 1), name="mask")

    ones_x = tf.ones_like(xin)
    x = tf.concat([xin, ones_x*mask], axis=3)
    # In original codes is: for the reason of indicating image bound?
    #x = tf.concat([xin, ones_x, ones_x*mask], axis=3)

    cnum = 48
    padding = 'same'
    x = gen_conv(x, cnum, 5, (1,1), padding, activation="elu", name='conv1')
    x = gen_conv(x, cnum*2, 3, (2,2), padding, activation="elu", name='conv2_ds')
    x = gen_conv(x, cnum*2, 3, (1,1), padding, activation="elu", name='conv3')
    x = gen_conv(x, cnum*4, 3, (2,2), padding, activation="elu", name='conv4_ds')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, activation="elu", name='conv5')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, activation="elu", name='conv6')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, dilation_rate=(2,2), activation="elu", name='conv7_astrous')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, dilation_rate=(4,4), activation="elu", name='conv8_astrous')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, dilation_rate=(8,8), activation="elu", name='conv9_astrous')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, dilation_rate=(16,16), activation="elu", name='conv10_astrous')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, activation="elu", name='conv11')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, activation="elu", name='conv12')
    x = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name='up1')(x)
    x = gen_conv(x, cnum*2, 3, (1,1), padding, activation="elu", name='conv13')
    x = gen_conv(x, cnum*2, 3, (1,1), padding, activation="elu", name='conv14')
    x = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name='up2')(x)
    x = gen_conv(x, cnum, 3, (1,1), padding, activation="elu", name='conv15')
    x = gen_conv(x, cnum//2, 3, (1,1), padding, activation="elu", name='conv16')
    x = layers.Conv2D(1, 3, (1,1), padding, activation="tanh", name='conv17')(x)
    #x = SNConv2D(1, 3, (1,1), padding, activation="tanh", name='conv17')(x)
    x_stage1 = x

    outputs = [x_stage1]

    coarse_inpaint_net = keras.Model(inputs=[xin, mask], outputs=outputs, name='coarse_inpaint_net')
    optimizer = keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.999)

    return coarse_inpaint_net, optimizer
