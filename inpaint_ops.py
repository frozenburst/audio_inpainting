import numpy as np
import tensorflow as tf
import math

from PIL import Image, ImageDraw


def mag_mel_weighted_map(loss_matrix):
    if len(loss_matrix.shape) == 4:
        # loss_matrix: [B, H, W, C]
        _, H, W, _ = loss_matrix.shape
    elif len(loss_matrix.shape) == 2:
        H, W = loss_matrix.shape
    else:
        raise ValueError("Unexpected shape of loss_matrix", loss_matrix.shape)
    w = tf.linspace(tf.ones(W), tf.ones(W)*10.0, H, axis=0)
    w = 1.-tf.experimental.numpy.log10(w)
    if len(loss_matrix.shape) == 4:
        w = tf.reshape(w, [1, H, W, 1])
    else:
        w = tf.reshape(w, [H, W])
    ones_w = tf.ones_like(w)
    scaler = tf.reduce_sum(ones_w) / tf.reduce_sum(w)
    return loss_matrix * w * scaler


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


def brush_stroke_mask(hp, name='mask'):
    """Generate mask tensor from bbox.

    Returns:
        tf.Tensor: output with shape [1, H, W, 1]

    """
    min_num_vertex = 4
    max_num_vertex = 12
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 15
    min_width = 12
    max_width = 40
    def generate_mask(H, W):
        average_radius = math.sqrt(H*H+W*W) / 8
        mask = Image.new('L', (W, H), 0)

        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                              v[1] - width//2,
                              v[0] + width//2,
                              v[1] + width//2),
                             fill=1)

        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.float32)
        mask = np.reshape(mask, (1, H, W, 1))
        return mask
    height = hp.image_height
    width = hp.image_width
    mask = tf.py_function(
            func=generate_mask,
            inp=[height, width],
            Tout=tf.float32)
    mask.set_shape([1] + [height, width] + [1])
    return mask
