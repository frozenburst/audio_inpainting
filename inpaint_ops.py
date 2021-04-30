import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import math

from tensorflow.keras import layers
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.keras.legacy_tf_layers import base
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import activations
from tensorflow.python.ops import init_ops
from sn_layers import SNConv2D

from summary_ops import scalar_summary
from PIL import Image, ImageDraw


# Reference from github: https://github.com/JiahuiYu/generative_inpainting
def gen_conv(x, cnum, ksize, stride=(1,1), padding='same',
             dilation_rate=(1,1), activation="elu", name='conv', sn=True, training=True):
    """Define conv for generator."""
    if sn:
        x = SNConv2D(cnum, ksize, stride, padding, dilation_rate=dilation_rate, activation=None, name=name)(x)
    else:
        x = layers.Conv2D(cnum, ksize, stride, padding, dilation_rate=dilation_rate, activation=None, name=name)(x)

    #if cnum == 3 or activation is None:
    #    # conv for output
    #    return x

    x, y = tf.split(x, 2, 3)

    if activation == "elu":
        x = tf.keras.activations.elu(x)
    elif activation == "relu":
        x = tf.keras.activations.relu(x)
    else:
        raise ValueError("Unknown activations: {activation}")
    #return x

    y = tf.keras.activations.sigmoid(y)
    x = x * y
    return x


def dis_conv(x, cnum, ksize=(5,5), strides=(2,2), padding='same', name='conv', training=True):
    """Define conv for discriminator.
    Activation is set to leaky_relu.
    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.
    Returns:
        tf.Tensor: output
    """
    x = SNConv2D(
            cnum, ksize, strides, padding, activation=None, name=name)(x)

    # default: 0.3, tf.nn.leaky_relu: 0.2
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x


def gan_hinge_loss(pos, neg, value=1., name='gan_hinge_loss'):
    """
    gan with hinge loss:
    https://github.com/pfnet-research/sngan_projection/blob/c26cedf7384c9776bcbe5764cb5ca5376e762007/updater.py
    """
    hinge_pos = tf.math.reduce_mean(tf.nn.relu(1.-pos))
    hinge_neg = tf.math.reduce_mean(tf.nn.relu(1.+neg))

    d_loss = tf.math.add(.5 * hinge_pos, .5 * hinge_neg)
    g_loss = -tf.math.reduce_mean(neg)
    return g_loss, d_loss, hinge_pos, hinge_neg
    # return g_loss, d_loss


def mag_mel_weighted_map(loss_matrix):
    # loss_matrix: [B, H, W, C]
    _, H, W, _ = loss_matrix.shape
    w = tf.linspace(tf.ones(W), tf.ones(W)*10.0, H, axis=0)
    w = 1.-tf.experimental.numpy.log10(w)
    w = tf.reshape(w, [1, H, W, 1])
    return loss_matrix * w


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


def split(tensor):
    new_list = []
    for i in range(tensor.shape[0]):
        t[i] = t[i][np.newaxis, :, :, :]
        new_list.append(t[i])
    return new_list


def contextual_attention(f, b, mask=None, ksize=3, stride=1, rate=1,
                         fuse_k=3, softmax_scale=10., training=True, fuse=True, batch_size=32):
    """ Contextual attention layer implementation.
    Contextual attention is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.
    Args:
        x: Input feature to match (foreground).
        b: Input feature for match (background).
        mask: Input mask for t, indicating patches not available.
        ksize: Kernel size for contextual attention.
        stride: Stride for extracting patches from t.
        rate: Dilation for matching.
        softmax_scale: Scaled softmax for attention.
        training: Indicating if current graph is training or inference.
    Returns:
        tf.Tensor: output
    """
    # get shapes
    # [b, h, w, c], [16, 64, 64, 96], [batch size, 256/stride, , cnum*4]
    raw_fs = tf.shape(f)
    raw_int_fs = f.get_shape().as_list()
    raw_int_bs = b.get_shape().as_list()
    # extract patches from background with stride and rate
    kernel = 2*rate
    # [16, 32, 32, 1536(96*16)] :extract 4*4 with stride=2 from h(64)*w(64) to be 32*32,
    # all patchs are stacked in the last, actually is 32*32(total patches) *16(each patch4*4).
    # raw_w = tf.extract_image_patches(
    #    b, [1,kernel,kernel,1], [1,rate*stride,rate*stride,1], [1,1,1,1], padding='SAME')
    raw_w = tf.image.extract_patches(
        b, [1,kernel,kernel,1], [1,rate*stride,rate*stride,1], [1,1,1,1], padding='SAME')
    #raw_w = tf.zeros([batch_size, 32, 32, 1536])
    # [16, 1024, 4, 4, 96]: 1024(total patches) * 4*4(each patch)
    # raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
    raw_w = tf.reshape(raw_w, [tf.shape(b)[0], -1, kernel, kernel, raw_int_bs[3]])
    # [16, 4, 4, 96, 1024]
    raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # downscaling foreground option: downscaling both foreground and
    # background for matching and use original background for reconstruction.
    # f, b, mask's h&w resize to half(rate=2)
    # f = resize(f, scale=1./rate, func=tf.image.resize_nearest_neighbor)
    f = layers.experimental.preprocessing.Resizing(int(raw_int_fs[1]/rate), int(raw_int_fs[2]/rate), 'nearest')(f)
    b = layers.experimental.preprocessing.Resizing(int(raw_int_bs[1]/rate), int(raw_int_bs[2]/rate), 'nearest')(b)
    # b = resize(b, to_shape=[int(raw_int_bs[1]/rate), int(raw_int_bs[2]/rate)], func=tf.image.resize_nearest_neighbor)  # https://github.com/tensorflow/tensorflow/issues/11651
    if mask is not None:
        # mask = resize(mask, scale=1./rate, func=tf.image.resize_nearest_neighbor)
        mask_shape = mask.get_shape().as_list()
        mask = layers.experimental.preprocessing.Resizing(int(mask_shape[1]/rate), int(mask_shape[2]/rate), 'nearest')(mask)
    # get shape after resize
    fs = tf.shape(f)
    int_fs = f.get_shape().as_list()
    # split f with each batch -> 16 * [1, 32, 32, 96] to group
    #f_groups = tf.split(f, int_fs[0], axis=0)
    # fix to batch_size should skip last batch
    f_groups = tf.split(f, batch_size, axis=0)
    # from t(H*W*C) to w(b*k*k*c*h*w)
    bs = tf.shape(b)
    int_bs = b.get_shape().as_list()
    # w: [16, 32, 32, 864(96*9)]: extract each patch(3*3) with stride=1 from b after resize
    # w = tf.extract_image_patches(
    #    b, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    w = tf.image.extract_patches(
        b, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    # [16, 1024(h*w), 3, 3, 96]
    w = tf.reshape(w, [tf.shape(f)[0], -1, ksize, ksize, int_fs[3]])
    # [16, 3, 3, 96, 1024]
    w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # process mask
    # [1, 32, 32, 1]
    if mask is None:
        mask = tf.zeros([1, int_bs[1], int_bs[2], 1])
    # m: [1, 32, 32, 9]
    m = tf.image.extract_patches(
        mask, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    # [1, 1024, 3, 3, 1]
    m = tf.reshape(m, [1, -1, ksize, ksize, 1])
    # [1, 3, 3, 1, 1024]
    m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    m = m[0]
    # mm: [1, 1, 1, 1024]:
    # reduce mean count the mean of seleted axis,
    # equal return True or False by condition, cast convert True->1 False->0.
    mm = tf.cast(tf.math.equal(tf.math.reduce_mean(m, axis=[0,1,2], keepdims=True), 0.), tf.float32)
    w_groups = tf.py_function(split, inp=[w], Tout=tf.float32)
    raw_w_groups = tf.py_function(split, inp=[raw_w], Tout=tf.float32)
    # 16 * [1, 3, 3, 96, 1024]: after resize
    w_groups = tf.split(w, batch_size, axis=0)
    # 16 * [1, 4, 4, 96, 1024]: original size
    raw_w_groups = tf.split(raw_w, batch_size, axis=0)
    y = []
    offsets = []
    k = fuse_k  # 3
    scale = softmax_scale   # 10.0
    # [3, 3, 1, 1]; 1, 0, 0 / 0, 1, 0 / 0, 0, 1; fuse_weight[0, 0, 0, 0] = 1
    fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
    for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
        # conv for compare
        wi = wi[0]
        wi_normed = wi / tf.math.maximum(tf.math.sqrt(tf.math.reduce_sum(tf.math.square(wi), axis=[0,1,2])), 1e-4)
        # [1, 32, 32, 1024]
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1,1,1,1], padding="SAME")
        # conv implementation for fuse scores to encourage large patches
        if fuse:
            # [1, 1024, 1024, 1]
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            # [1, 32, 32, 32, 32]
            yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            # [1, 1024, 1024, 1]
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            # [1, 32, 32, 32, 32]
            yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
        # [1, 32, 32, 1024]
        yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1]*bs[2]])

        # softmax to match
        yi *=  mm  # mask
        yi = tf.nn.softmax(yi*scale, 3)
        yi *=  mm  # mask

        # offset: [1, 32, 32]
        offset = tf.math.argmax(yi, axis=3, output_type=tf.int32)
        # [1, 32, 32, 2]
        offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)
        # deconv for patch pasting
        # 3.1 paste center
        # [4, 4, 96, 1024]
        wi_center = raw_wi[0]
        # [1, 64, 64, 96]
        yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0), strides=[1,rate,rate,1]) / 4.
        y.append(yi)
        offsets.append(offset)
    # 16 * [1, 64, 64, 96]
    y = tf.concat(y, axis=0)
    y.set_shape(raw_int_fs)
    # [16, 32, 32, 2]
    offsets = tf.concat(offsets, axis=0)
    offsets.set_shape(int_bs[:3] + [2])
    # case1: visualize optical flow: minus current position
    # [16, 32, 32, 1]
    h_add = tf.tile(tf.reshape(tf.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
    w_add = tf.tile(tf.reshape(tf.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
    offsets = offsets - tf.concat([h_add, w_add], axis=3)
    # to flow image
    # [16, 32, 32, 1]
    flow = flow_to_image_tf(offsets)
    # # case2: visualize which pixels are attended
    flow_2 = highlight_flow_tf(offsets * tf.cast(mask, tf.int32))
    if rate != 1:
        # [16, 64, 64, 1]
        # flow = resize(flow, scale=rate, func=tf.image.resize_bilinear)
        # flow shape
        #breakpoint()
        #flow = tf.zeros([batch_size, 32, 32, 1])
        #flow_2 = tf.zeros([batch_size, 32, 32, 1])
        flow_shape = flow.get_shape().as_list()
        flow = layers.experimental.preprocessing.Resizing(
                int(flow_shape[1]*rate), int(flow_shape[2]*rate), 'bilinear')(flow)

        #flow_2_shape = flow_2.get_shape().as_list()
        #flow_2 = layers.experimental.preprocessing.Resizing(
        #            int(flow_2_shape[1]*rate), int(flow_2_shape[2]*rate), 'bilinear')(flow_2)
    return y, flow, flow_2


def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


COLORWHEEL = make_color_wheel()


def compute_color(u,v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u**2+v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))
    return img



def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u/(maxrad + np.finfo(float).eps)
        v = v/(maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))


def flow_to_image_tf(flow, name='flow_to_image'):
    """Tensorflow ops for computing flow to image.
    """
    img = tf.py_function(flow_to_image, inp=[flow], Tout=tf.float32)
    img.set_shape(flow.get_shape().as_list()[0:-1]+[1])
    img = img / 127.5 - 1. # Here is the compute for the flow color
    return img


def highlight_flow(flow):
    """Convert flow into middlebury color code image.
    """
    out = []
    s = flow.shape
    for i in range(flow.shape[0]):
        img = np.ones((s[1], s[2], 3)) * 144.
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        for h in range(s[1]):
            for w in range(s[1]):
                ui = u[h,w]
                vi = v[h,w]
                img[ui, vi, :] = 255.
        out.append(img)
    return np.float32(np.uint8(out))


def highlight_flow_tf(flow, name='flow_to_image'):
    """Tensorflow ops for highlight flow.
    """
    img = tf.py_function(highlight_flow, inp=[flow], Tout=tf.float32)
    img.set_shape(flow.get_shape().as_list()[0:-1]+[1])
    img = img / 127.5 - 1. # Normalize flow color
    return img
