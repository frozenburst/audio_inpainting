from data_loader import *
from models import *
from tensorflow import keras

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib

print(tf.__version__)

class hp:
    # parameter
    training_file = './data/train_list.txt'
    testing_file = './data/test_list.txt'
    logdir = './logs/spec'
    #checkpoint_dir = './checkpoints/th_spec'
    epochs = 1
    validation_split = 0.1
    num_classes = 50
    batch = True
    batch_size = 32
    image_height = 256
    image_width = 256


def write_predict_output(epoch, logs, color_format='GRAY', max_outs=3):
    #test_data = tf.data.experimental.sample_from_datasets(test_dataset)
    test_spec = model.predict(test_data)
    #print(test_data[:max_outs].shape, test_data[:max_outs].max(), test_data[:max_outs].min())
    #print(test_specs.shape, test_specs.max(), test_specs.min())
    img = [test_data, test_spec]
    img_cat = tf.concat(img, axis=2)
    img_cat = (img_cat + 1.) / 2.
    #print(img_cat.shape)
    ## Makes -1~1 to 0~255
    #if color_format == 'GRAY':
    #    img_cat = tf.clip_by_value((img_cat+1.)*127.5, 0, 255)

    with file_writer_img.as_default():
        tf.summary.image("Test output", img_cat, step=epoch, max_outputs=1)
        #tf.summary.image("Test output", test_specs, step=epoch, max_outputs=max_outs)

def load_pyfunc(filename):
    print(filename.numpy())
    breakpoint()
    return 2.64


if __name__ == "__main__":

    # load data
    print("Load data from path...")
    train_data_fnames, train_labels = load_data_filename(hp.training_file)
    test_data_fnames, test_labels = load_data_filename(hp.testing_file)
    print(train_data_fnames.shape, test_data_fnames.shape)
    print(train_data_fnames[0])

    print("Prefetching...")
    # seems can not work on TF 1.x
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data_fnames))
    train_dataset = train_dataset.map(lambda x: tf.py_function(load_npy, inp=[x], Tout=[tf.float32, tf.float32]))
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(hp.batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_data_fnames))
    test_dataset = test_dataset.map(lambda x: tf.py_function(load_npy, inp=[x], Tout=[tf.float32, tf.float32]))
    test_dataset = test_dataset.batch(hp.batch_size)

    #print(f'training list\'s shape:{train_data.shape}, testing list\'s shape: {test_data.shape}')

    # preprocess
    #train_labels = keras.utils.to_categorical(train_labels)
    #test_labels = keras.utils.to_categorical(test_labels)

    #import pdb; pdb.set_trace()
    
    print("Build model...")
    model = auto_encoder(hp.image_height, hp.image_width)

    

    tensorboard_callbacks = tf.keras.callbacks.TensorBoard(
            log_dir=hp.logdir, histogram_freq=1)
    file_writer_img = tf.summary.create_file_writer(hp.logdir + '/img')

    #img_callback = keras.callbacks.LambdaCallback(on_epoch_end=write_predict_output)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "./checkpoints/spec/save_at_{epoch}_best.h5",
            monitor='val_loss',
            mode='min',
            save_best_only=True),
        tensorboard_callbacks,
        #img_callback,
    ]

    print(model.summary())

    model.fit(train_dataset, epochs=hp.epochs,
              validation_data=test_dataset, validation_freq=2, callbacks=callbacks)


    test_loss, test_mse = model.evaluate(test_dataset, verbose=2)
    print('\nTest error:', test_mse)
