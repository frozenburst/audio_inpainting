from data_loader import *
from models import *
from tensorflow import keras

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib

print(tf.__version__)


def write_predict_output(epoch, logs, color_format='GRAY', max_outs=3):
    test_specs = model.predict(test_data[:max_outs])
    #print(test_data[:max_outs].shape, test_data[:max_outs].max(), test_data[:max_outs].min())
    #print(test_specs.shape, test_specs.max(), test_specs.min())
    img = [test_data[:max_outs], test_specs]
    img_cat = tf.concat(img, axis=2)
    img_cat = (img_cat + 1.) / 2.
    #print(img_cat.shape)
    ## Makes -1~1 to 0~255
    #if color_format == 'GRAY':
    #    img_cat = tf.clip_by_value((img_cat+1.)*127.5, 0, 255)

    with file_writer_img.as_default():
        tf.summary.image("Test output", img_cat, step=epoch, max_outputs=max_outs)
        #tf.summary.image("Test output", test_specs, step=epoch, max_outputs=max_outs)


if __name__ == "__main__":
    # parameter
    training_file = './data/th_train_list.txt'
    testing_file = './data/th_test_list.txt'
    logdir = './logs/th_spec'
    #checkpoint_dir = './checkpoints/th_spec'
    epochs = 50
    validation_split = 0.1
    num_classes = 50
    batch = True
    batch_size = 32
    image_height = 256
    image_width = 256

    # load data
    print("Load data from path...")
    train_data, train_labels = load_data_filename(training_file, batch=batch)
    test_data, test_labels = load_data_filename(testing_file, batch=batch)
    print(train_data.shape, test_data.shape)
    
    print("Prefetching...")
    # seems can not work on TF 1.x
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_data))
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    print(f'training list\'s shape:{train_data.shape}, testing list\'s shape: {test_data.shape}')

    # preprocess
    #train_labels = keras.utils.to_categorical(train_labels)
    #test_labels = keras.utils.to_categorical(test_labels)

    #import pdb; pdb.set_trace()
    
    print("Build model...")
    model = auto_encoder(image_height, image_width)

    

    tensorboard_callbacks = tf.keras.callbacks.TensorBoard(
            log_dir=logdir, histogram_freq=1)
    file_writer_img = tf.summary.create_file_writer(logdir + '/img')

    img_callback = keras.callbacks.LambdaCallback(on_epoch_end=write_predict_output)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "./checkpoints/th_spec/save_at_{epoch}_best.h5",
            monitor='val_loss',
            mode='min',
            save_best_only=True),
        tensorboard_callbacks,
        img_callback,
    ]

    print(model.summary())

    model.fit(train_dataset, epochs=epochs,
              validation_data=(test_data, test_data), validation_freq=2, callbacks=callbacks)

    test_loss, test_mse = model.evaluate(test_data, test_data, verbose=2)
    print('\nTest error:', test_mse)
