from data_loader import *
from models import *
from tensorflow.keras.optimizers import SGD
from tensorflow import keras

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib

print(tf.__version__)

if __name__ == "__main__":
    # parameter
    training_file = './data/train_list.txt'
    testing_file = './data/test_list.txt'
    epochs = 10
    validation_split = 0.1
    num_classes = 50
    batch = True
    batch_size = 32
    image_height = 256
    image_width = 256

    # load data
    train_data, train_labels = load_data_filename(training_file, batch=batch)
    test_data, test_labels = load_data_filename(testing_file, batch=batch)

    # seems can not work on TF 1.x
    #train_data = train_data.prefetch(buffer_size=32)

    print(f'training list\'s shape:{train_data.shape}, testing list\'s shape: {test_data.shape}')

    # preprocess
    train_labels = keras.utils.to_categorical(train_labels)
    test_labels = keras.utils.to_categorical(test_labels)

    #import pdb; pdb.set_trace()

    model = xception(num_classes, image_height, image_width)


    callbacks = [
        keras.callbacks.ModelCheckpoint("./checkpoints/save_at_{epoch}.h5"),
    ]

    print(model.summary())

    model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs,
              validation_split=0.1, validation_freq=2, callbacks=callbacks)

    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
