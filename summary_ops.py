import tensorflow as tf
import os.path as op


def scalar_summary(name, value, step=None):
    tf.summary.scalar(name, value, step=step)


def dict_scalar_summary(name, _dict, step=None):
    if _dict is None:
        raise ValueError("None in the dictionary.")

    for key in _dict.keys():
        tf.summary.scalar(op.join(name, key), _dict[key], step=step)


def images_summary(name, value, step=None, max_outputs=0, color_format='None'):
    # Make -1 ~ 1 -> 0 ~ 1
    value = value[:max_outputs]
    value = (value + 1.) / 2.

    tf.summary.image(name, value, step=step, max_outputs=max_outputs)


def gradient_calc(y, x, norm=tf.math.abs):
    grad = tf.math.reduce_sum(norm(tf.gradients(y, x)))
    #scalar_summary(name, grad, step)
    return grad


def audio_summary(name, waveform, sample_rate, step, max_outputs):
    tf.summary.audio(name, waveform, sample_rate, step=step, max_outputs=max_outputs)
