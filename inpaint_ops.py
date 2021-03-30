import numpy as np
import tensorflow as tf

from tensorflow.keras import layers

# Reference from github: https://github.com/JiahuiYu/generative_inpainting
def gen_conv(x, cnum, ksize, stride=(1,1), padding='same',
             dilation_rate=(1,1), activation="elu", name='conv', training=True):
    """Define conv for generator."""
    x = layers.Conv2D(cnum, ksize, stride, padding, dilation_rate=dilation_rate, activation=None, name=name)(x)
    if cnum == 3 or activation is None:
        # conv for output
        return x
    x, y = tf.split(x, 2, 3)

    if activation == "elu":
        x = tf.keras.activations.elu(x)
    else:
        raise ValueError("Unknown activations: {activation}")

    y = tf.keras.activations.sigmoid(y)
    x = x * y
    return x


def random_bbox(hp):
    """Generate a random tlhw.
    Returns:
        tuple: (top, left, height, width)
    """
    img_height = hp.image_height
    img_width = hp.image_width
    # The author from deepfill version seems to be wrong.
    # maxt = img_height - FLAGS.vertical_margin - FLAGS.height
    # maxl = img_width - FLAGS.horizontal_margin - FLAGS.width

    # +1 for exclusive of max bound.
    maxt = img_height - hp.mask_height + 1
    maxl = img_width - hp.mask_width + 1
    t = tf.random.uniform(
        [], minval=hp.vertical_margin, maxval=maxt, dtype=tf.int32)
    l = tf.random.uniform(
        [], minval=hp.horizontal_margin, maxval=maxl, dtype=tf.int32)
    h = tf.constant(hp.mask_height)
    w = tf.constant(hp.mask_width)
    return (t, l, h, w)


def bbox2mask(hp, bbox, name='mask'):
    """Generate mask tensor from bbox.
    Args:
        bbox: tuple, (top, left, height, width)
    Returns:
        tf.Tensor: output with shape [1, H, W, 1]
    """
    def npmask(bbox, height, width, delta_h, delta_w):
        mask = np.zeros((1, height, width, 1), np.float32)
        h = np.random.randint(delta_h//2+1)
        w = np.random.randint(delta_w//2+1)
        mask[:, bbox[0]+h:bbox[0]+bbox[2]-h+1,
            bbox[1]+w:bbox[1]+bbox[3]-w+1, :] = 1.
        return mask

    height = hp.image_height
    width = hp.image_width
    mask = tf.py_function(
                func=npmask,
                inp=[bbox, height, width,
                     hp.max_delta_height, hp.max_delta_width],
                Tout=tf.float32)
    mask.set_shape([1] + [height, width] + [1])
    return mask
