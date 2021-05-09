import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from inpaint_ops import gen_conv
from sn_layers import SNConv2D


def classifier(img_height=256, img_width=256, img_c=1, training=True, name='C'):

    x_mask = keras.Input(shape=(img_height, img_width, img_c), name="x_mask")
    #pos_mask = keras.Input(shape=(img_height, img_width, img_c), name="x_mask")
    x_unmask = keras.Input(shape=(img_height, img_width, img_c), name="x_unmask")
    mask = keras.Input(shape=(img_height, img_width, img_c), name="mask")

    inputs = [x_mask, x_unmask, mask]

    ones_x = tf.ones_like(xin)
    x_m = tf.concat([x_mask, ones_x*mask], axis=-1)
    x_u = tf.concat([x_unmask, ones_x*(1. - mask)], axis=-1)
    x = tf.concat([x_m, x_u], axis=0)

    cnum = 64
    padding = 'same'
    x = gen_conv(x, cnum, 3, (1,1), padding, activation="elu", name='conv1')
    x = gen_conv(x, cnum*2, 3, (2,2), padding, activation="elu", name='conv2_ds')
    x = gen_conv(x, cnum*2, 3, (1,1), padding, activation="elu", name='conv3')
    x = gen_conv(x, cnum*4, 3, (2,2), padding, activation="elu", name='conv4_ds')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, activation="elu", name='conv5')
    x = gen_conv(x, cnum*8, 3, (2,2), padding, activation="elu", name='conv6_ds')
    x = gen_conv(x, cnum*8, 3, (1,1), padding, dilation_rate=(2,2), activation="elu", name='conv7_astrous')
    x = gen_conv(x, cnum*8, 3, (1,1), padding, dilation_rate=(4,4), activation="elu", name='conv8_astrous')
    x = gen_conv(x, cnum*8, 3, (1,1), padding, dilation_rate=(8,8), activation="elu", name='conv9_astrous')
    x = gen_conv(x, cnum*8, 3, (1,1), padding, dilation_rate=(16,16), activation="elu", name='conv10_astrous')
    x = gen_conv(x, cnum*8, 3, (1,1), padding, activation="elu", name='conv11')
    x = gen_conv(x, cnum*8, 3, (1,1), padding, activation="elu", name='conv12')
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, activation="sigmoid")(x)

    outputs = x

    classifier = keras.Model(inputs=inputs, outputs=outputs, name='C')
    #optimizer = keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.999)
    return classifier
