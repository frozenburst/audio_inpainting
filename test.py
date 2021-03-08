from data_loader import *
from models import *
from tensorflow.keras.optimizers import SGD
from tensorflow import keras
from tensorflow.keras.models import Model

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib

print(tf.__version__)

if __name__ == "__main__":
    # parameter
    training_file = './data/train_list.txt'
    testing_file = './data/test_list.txt'
    saved_model_pth = './saved_models'
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

    model = xception(num_classes, image_height, image_width, training=False)

    print(model.summary())
    model.load_weights('./checkpoints/xception_01/save_at_10.h5')

    layer_name = 'global_average_pooling2d'
    #layer_name = 'dense'
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_layer_model.compile(
                 optimizer=keras.optimizers.Adam(1e-3),
                 loss="binary_crossentropy",
                 metrics=["accuracy"],
    )
    intermediate_output = intermediate_layer_model.predict(test_data)
    print(intermediate_output.shape)
    print(intermediate_output)


    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    intermediate_layer_model.save(f'{saved_model_pth}/inter_model.h5')

    reconstructed_model = keras.models.load_model(f'{saved_model_pth}/inter_model.h5')

    np.testing.assert_allclose(
        intermediate_layer_model.predict(test_data), reconstructed_model.predict(test_data)
    )

    print(reconstructed_model.summary())
    print(test_data.shape)
    reconstructed_model_output = reconstructed_model.predict(test_data)
    print(reconstructed_model_output.shape, reconstructed_model_output)
