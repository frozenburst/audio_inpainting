import numpy as np
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

from inpaint_ops import gen_conv


def one_dense_layer(num_classes=50):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(256, 256)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def conv_layers(num_classes=50, img_height=256, img_width=256):
    model = Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 1)),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def vgg(num_classes=50, img_height=256, img_width=256):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


# from https://keras.io/examples/vision/image_classification_from_scratch/
def xception(num_classes=50, img_height=256, img_width=256, training=True):
    inputs = keras.Input(shape=(img_height, img_width, 1))
    # Image augmentation block
    #x = data_augmentation(inputs)
    x = inputs

    # Entry block
    #x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same", trainable=training)(x)
    x = layers.BatchNormalization(trainable=training)(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same", trainable=training)(x)
    x = layers.BatchNormalization(trainable=training)(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same", trainable=training)(x)
        x = layers.BatchNormalization(trainable=training)(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same", trainable=training)(x)
        x = layers.BatchNormalization(trainable=training)(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same", trainable=training)(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same", trainable=training)(x)
    x = layers.BatchNormalization(trainable=training)(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5, trainable=training)(x)
    outputs = layers.Dense(units, activation=activation, trainable=training)(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def auto_encoder(img_height=256, img_width=256, training=True):

    inputs = keras.Input(shape=(img_height, img_width, 2), name="spec")
    xin, mask = tf.split(inputs, num_or_size_splits=2, axis=3)

    cnum = 48
    padding = 'same'
    x = gen_conv(inputs, cnum, 3, (1,1), padding, activation="elu", name='conv1')
    x = gen_conv(x, cnum*2, 3, (2,2), padding, activation="elu", name='conv2_ds')
    x = gen_conv(x, cnum*2, 3, (1,1), padding, activation="elu", name='conv3')
    x = gen_conv(x, cnum*4, 3, (2,2), padding, activation="elu", name='conv4_ds')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, activation="elu", name='conv5')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, activation="elu", name='conv6')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, dilation_rate=(2,2), activation="elu", name='conv7')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, dilation_rate=(4,4), activation="elu", name='conv8')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, dilation_rate=(8,8), activation="elu", name='conv9')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, dilation_rate=(16,16), activation="elu", name='conv10')
    x = gen_conv(x, cnum*4, 3, (1,1), padding, activation="elu", name='conv11')
    x = layers.Conv2D(cnum*4, 3, (1,1), padding, activation="tanh", name='conv12')(x)
    encoder_output = x

    x = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name='up1')(x)
    x = gen_conv(x, cnum*2, 3, (1,1), padding, activation="elu", name='conv13')
    x = gen_conv(x, cnum*2, 3, (1,1), padding, activation="elu", name='conv14')
    x = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name='up2')(x)
    x = gen_conv(x, cnum, 3, (1,1), padding, activation="elu", name='conv15')
    x = gen_conv(x, cnum/2, 3, (1,1), padding, activation="elu", name='conv16')
    x = layers.Conv2D(1, 3, (1,1), padding, activation="tanh", name='conv17')(x)
    decoder_output = x

    autoencoder = keras.Model(inputs, decoder_output, name="autoencoder")

    # self train loop should comduct loss within loop. So no auto reduction here.
    #loss_fn = tf.keras.losses.MeanAbsoluteError(reduction='sum_over_batch_size')
    loss_fn = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

    optimizer = keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.999)
    ''' for keras model fit
    autoencoder.compile(
    	optimizer=keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.999),
        loss=loss_fn,
        metrics=["mean_absolute_error"],
    )
    '''
    return autoencoder, optimizer, loss_fn
