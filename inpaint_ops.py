import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import layers

# Reference from github: https://github.com/JiahuiYu/generative_inpainting
def gen_conv(x, cnum, ksize, stride=(1,1), padding='same',
             dilation_rate=(1,1), activation="elu", name='conv', training=True):
    """Define conv for generator."""
    x = layers.Conv2D(cnum, ksize, stride, padding, dilation_rate=dilation_rate, activation=None, name=name)(x)

    if cnum == 3 or activation is None:
        # conv for output
        return x

    #x, y = tf.split(x, 2, 3)

    if activation == "elu":
        x = tf.keras.activations.elu(x)
    elif activation == "relu":
        x = tf.keras.activations.relu(x)
    else:
        raise ValueError("Unknown activations: {activation}")
    return x

    #y = tf.keras.activations.sigmoid(y)
    #x = x * y
    #return x


def kernel_spectral_norm(kernel, iteration=1, name='kernel_sn'):
    # spectral_norm
    def l2_norm(input_x, epsilon=1e-12):
        input_x_norm = input_x / (tf.math.reduce_sum(input_x**2)**0.5 + epsilon)
        return input_x_norm
    w_shape = kernel.get_shape().as_list()
    w_mat = tf.reshape(kernel, [-1, w_shape[-1]])
    u = tf.compat.v1.get_variable(
        'u', shape=[1, w_shape[-1]],
        initializer=tf.compat.v1.truncated_normal_initializer(),
        trainable=False)

    def power_iteration(u, ite):
        v_ = tf.linalg.matmul(u, tf.transpose(w_mat))
        v_hat = l2_norm(v_)
        u_ = tf.linalg.matmul(v_hat, w_mat)
        u_hat = l2_norm(u_)
        return u_hat, v_hat, ite+1

    u_hat, v_hat,_ = power_iteration(u, iteration)
    sigma = tf.matmul(tf.linalg.matmul(v_hat, w_mat), tf.transpose(u_hat))
    w_mat = w_mat / sigma
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_mat, w_shape)
    return w_norm


class Conv2DSepctralNorm(tf.keras.layers.Conv2D):
    def build(self, input_shape):
        super(Conv2DSepctralNorm, self).build(input_shape)
        self.kernel = kernel_spectral_norm(self.kernel)


def conv2d_spectral_norm(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None):
    layer = Conv2DSepctralNorm(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name,
        dtype=inputs.dtype.base_dtype)
    return layer.apply(inputs)


# Reference from https://github.com/thisisiron/spectral_normalization-tf2/blob/master/sn.py
class SpectralNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        self.iteration = iteration
        self.eps = eps
        self.do_power_iteration = training
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.v = self.add_weight(shape=(1, self.w_shape[0] * self.w_shape[1] * self.w_shape[2]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name='sn_v',
                                 dtype=tf.float32)

        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name='sn_u',
                                 dtype=tf.float32)

        super(SpectralNormalization, self).build()

    def call(self, inputs):
        self.update_weights()
        output = self.layer(inputs)
        self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
        return output

    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

        u_hat = self.u
        v_hat = self.v  # init v vector

        if self.do_power_iteration:
            for _ in range(self.iteration):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                v_hat = v_ / (tf.reduce_sum(v_**2)**0.5 + self.eps)

                u_ = tf.matmul(v_hat, w_reshaped)
                u_hat = u_ / (tf.reduce_sum(u_**2)**0.5 + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)
        self.v.assign(v_hat)

        self.layer.kernel.assign(self.w / sigma)

    def restore_weights(self):
        self.layer.kernel.assign(self.w)


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
    x = SpectralNormalization(
            layers.Conv2D(cnum, ksize, strides, padding, activation=None, name=name))(x)
    #x = tfa.layers.SpectralNormalization(
    #        layers.Conv2D(cnum, ksize, strides, padding, activation=None, name=name))(x)
    #x = layers.Conv2D(cnum, ksize, strides, padding, activation=None, name=name)(x)
    #x = conv2d_spectral_norm(x, cnum, ksize, strides, padding, name=name)

    x = layers.LeakyReLU()(x)
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
    return g_loss, d_loss


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
