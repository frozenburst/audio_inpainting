import tensorflow as tf

def scalar_summary(name, value, step=None):
    tf.summary.scalar(name, value, step=step)


def images_summary(name, value, step=None, max_outputs=0, color_format='None'):
    # Make -1 ~ 1 -> 0 ~ 1
    value = value[:max_outputs]
    value = (value + 1.) / 2.

    tf.summary.image(name, value, step=step, max_outputs=max_outputs)
