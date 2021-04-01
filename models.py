import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from inpaint_ops import gen_conv, dis_conv
from inpaint_ops import contextual_attention


# Reference from GitHub: https://github.com/JiahuiYu/generative_inpainting
def inpaint_net(img_height=256, img_width=256, batch_size=32, training=True):

    xin = keras.Input(shape=(img_height, img_width, 1), name="image")
    mask = keras.Input(shape=(img_height, img_width, 1), name="mask")
    offset_flow = None

    ones_x = tf.ones_like(xin)
    x = tf.concat([xin, ones_x*mask], axis=3)
    # In original codes is: for the reason of indicating image bound?
    # x = tf.concat([xin, ones_x, ones_x*mask], axis=3)
    #xin, mask = tf.split(inputs, num_or_size_splits=2, axis=3)

    cnum = 48
    padding = 'same'
    x = gen_conv(x, cnum, 5, (1,1), padding, activation="elu", name='conv1')
    x = gen_conv(x, cnum*2, 3, (2,2), padding, activation="elu", name='conv2_ds')
    x = gen_conv(x, cnum*2, 3, (1,1), padding, activation="elu", name='conv3')
    x = gen_conv(x, cnum*4, 3, (2,2), padding, activation="elu", name='conv4_ds')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, activation="elu", name='conv5')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, activation="elu", name='conv6')
    x_shape = x.get_shape().as_list()[1:3]
    mask_s = layers.experimental.preprocessing.Resizing(x_shape[0], x_shape[1], 'nearest')(mask)
    x = gen_conv(x, cnum*4, 3, (1,1), padding, dilation_rate=(2,2), activation="elu", name='conv7_astrous')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, dilation_rate=(4,4), activation="elu", name='conv8_astrous')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, dilation_rate=(8,8), activation="elu", name='conv9_astrous')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, dilation_rate=(16,16), activation="elu", name='conv10_astrous')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, activation="elu", name='conv11')
    x = layers.Conv2D(cnum*4, 3, (1,1), padding, activation="tanh", name='conv12')(x)

    x = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name='up1')(x)
    x = gen_conv(x, cnum*2, 3, (1,1), padding, activation="elu", name='conv13')
    x = gen_conv(x, cnum*2, 3, (1,1), padding, activation="elu", name='conv14')
    x = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name='up2')(x)
    x = gen_conv(x, cnum, 3, (1,1), padding, activation="elu", name='conv15')
    x = gen_conv(x, cnum//2, 3, (1,1), padding, activation="elu", name='conv16')
    x = layers.Conv2D(1, 3, (1,1), padding, activation="tanh", name='conv17')(x)
    x_stage1 = x

    # autoencoder = keras.Model(inputs, decoder_output, name="autoencoder")

    x = x*mask + xin*(1.-mask)
    x_stage2_input = x
    x = gen_conv(x_stage2_input, cnum, 5, (1,1), padding, activation="elu", name='xconv1')
    x = gen_conv(x, cnum, 3, (2,2), padding, activation="elu", name='xconv2_ds')
    x = gen_conv(x, cnum*2, 3, (1,1), padding, activation="elu", name='xconv3')
    x = gen_conv(x, cnum*2, 3, (2,2), padding, activation="elu", name='xconv4_ds')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, activation="elu", name='xconv5')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, activation="elu", name='xconv6')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, dilation_rate=(2,2), activation="elu", name='xconv7_astrous')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, dilation_rate=(4,4), activation="elu", name='xconv8_astrous')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, dilation_rate=(8,8), activation="elu", name='xconv9_astrous')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, dilation_rate=(16,16), activation="elu", name='xconv10_astrous')
    x_hallu = x
    # attention branch
    x = gen_conv(x_stage2_input, cnum, 5, (1,1), padding, activation="elu", name='pmconv1')
    x = gen_conv(x, cnum, 3, (2,2), padding, activation="elu", name='pmconv2_ds')
    x = gen_conv(x, cnum*2, 3, (1,1), padding, activation="elu", name='pmconv3')
    x = gen_conv(x, cnum*2, 3, (2,2), padding, activation="elu", name='pmconv4_ds')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, activation="elu", name='pmconv5')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, activation="relu", name='pmconv6')
    x, offset_flow, offset_flow_m = contextual_attention(f=x, b=x, mask=mask_s, ksize=3, stride=1, rate=2, batch_size=batch_size)
    # x = contextual_attention(f=x, b=x, mask=mask_s, ksize=3, stride=1, rate=2, batch_size=batch_size)
    x = gen_conv(x, cnum*4, 3, (1,1), padding, activation="elu", name='pmconv9')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, activation="elu", name='pmconv10')
    pm = x
    x = tf.concat([x_hallu, pm], axis=3)

    x = gen_conv(x, cnum*4, 3, (1,1), padding, activation="elu", name='allconv11')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, activation="elu", name='allconv12')
    x = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name='allup1')(x)
    x = gen_conv(x, cnum*2, 3, (1,1), padding, activation="elu", name='allconv13')
    x = gen_conv(x, cnum*2, 3, (1,1), padding, activation="elu", name='allconv14')
    x = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name='allup2')(x)
    x = gen_conv(x, cnum, 3, (1,1), padding, activation="elu", name='allconv15')
    x = gen_conv(x, cnum//2, 3, (1,1), padding, activation="elu", name='allconv16')
    x = layers.Conv2D(1, 3, (1,1), padding, activation="tanh", name='allconv17')(x)
    x_stage2 = x

    offset_flow.set_shape([batch_size, 64, 64, 1])

    #flow = layers.Lambda(tensor_wrapper)(offset_flow)
    #flow_m = layers.Lambda(tensor_wrapper)(offset_flow_m)
    #flow = layers.Lambda(lambda x: layers.Concatenate(axis=2)(x))([x_hallu, pm])

    #outputs = [x_stage1, x_stage2, x_stage2_input, flow]
    #outputs = [x_stage1, x_stage2, x_stage2_input, offset_flow, offset_flow_m]
    outputs = [x_stage1, x_stage2, x_stage2_input, offset_flow]
    inpaint_net = keras.Model(inputs=[xin, mask], outputs=outputs, name='inpaint_net')

    # self train loop should comduct loss within loop. So no auto reduction here.
    #loss_fn = tf.keras.losses.MeanAbsoluteError(reduction='sum_over_batch_size')
    #loss_fn = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

    #optimizer = keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.999)
    ''' for keras model fit
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.999),
        loss=loss_fn,
        metrics=["mean_absolute_error"],
    )
    '''
    return inpaint_net


def sn_patch_gan_discriminator(img_height=256, img_width=256, training=True):
    xin = keras.Input(shape=(img_height, img_width, 2), name='img')

    cnum = 64
    x = dis_conv(xin, cnum, name='disconv1', training=training)
    x = dis_conv(x, cnum*2, name='disconv2', training=training)
    x = dis_conv(x, cnum*4, name='disconv3', training=training)
    x = dis_conv(x, cnum*4, name='disconv4', training=training)
    x = dis_conv(x, cnum*4, name='disconv5', training=training)
    x = dis_conv(x, cnum*4, name='disconv6', training=training)
    x = layers.Flatten()(x)
    output = x

    sn_patch_gan_d = keras.Model(inputs=xin, outputs=output, name='sn_patch_gan_d')
    return sn_patch_gan_d
