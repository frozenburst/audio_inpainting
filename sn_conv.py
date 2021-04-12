# Reference from tensorflow github:
# https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/layers/convolutional.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import six
import tensorflow as tf

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
# imports for backwards namespace compatibility
# pylint: disable=unused-import
# pylint: enable=unused-import
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops


class SNConv(Layer):
    def __init__(self,
                 rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 groups=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 conv_op=None,
                 iteration=1,
                 eps=1e-12,
                 **kwargs):
        super(SNConv, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        self.rank = rank

        if isinstance(filters, float):
            filters = int(filters)
        self.filters = filters
        self.groups = groups or 1
        self.kernel_size = conv_utils.normalize_tuple(
                            kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
                                dilation_rate, rank, 'dilation_rate')

        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=self.rank + 2)

        self._validate_init()
        self._is_causal = self.padding == 'causal'
        self._channels_first = self.data_format == 'channels_first'
        self._tf_data_format = conv_utils.convert_data_format(
            self.data_format, self.rank + 2)

        ## For Spectral Norm
        self.iteration = iteration
        self.eps = eps

    def _validate_init(self):
        if self.filters is not None and self.filters % self.groups != 0:
            raise ValueError(
                'The number of filters must be evenly divisible by the number of '
                'groups. Received: groups={}, filters={}'.format(
                    self.groups, self.filters))

        if not all(self.kernel_size):
            raise ValueError('The argument `kernel_size` cannot contain 0(s). '
                       'Received: %s' % (self.kernel_size,))

        if (self.padding == 'causal' and not isinstance(self,
                                                    (tf.keras.layers.Conv1D, tf.keras.layers.SeparableConv1D))):
            raise ValueError('Causal padding is only supported for `Conv1D`'
                               'and `SeparableConv1D`.')

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        if input_channel % self.groups != 0:
           raise ValueError(
                'The number of input channels must be evenly divisible by the number '
                'of groups. Received groups={}, but the input has {} channels '
                '(full input shape is {}).'.format(self.groups, input_channel,
                                                 input_shape))
        kernel_shape = self.kernel_size + (input_channel // self.groups,
                                       self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                axes={channel_axis: input_channel})

        ## Spectral Norm
        '''
        self.u = self.add_weight(
                        shape=(1, self.kernel_shape[-1]),
                        initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                        trainable=False,
                        name='sn_u',
                        dtype=tf.float32)
        '''
        self.u = self.add_weight(
                        shape=(1, kernel_shape[-1]),
                        initializer=tf.keras.initializers.RandomNormal(),
                        trainable=False,
                        name='sn_u',
                        dtype=tf.float32
        )
        # Convert Keras formats to TF native formats.
        if self.padding == 'causal':
            tf_padding = 'VALID'  # Causal padding handled in `call`.
        elif isinstance(self.padding, six.string_types):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding
        tf_dilations = list(self.dilation_rate)
        tf_strides = list(self.strides)

        tf_op_name = self.__class__.__name__
        if tf_op_name == 'Conv1D':
            tf_op_name = 'conv1d'  # Backwards compat.

        self._convolution_op = functools.partial(
            nn_ops.convolution_v2,
            strides=tf_strides,
            padding=tf_padding,
            dilations=tf_dilations,
            data_format=self._tf_data_format,
            name=tf_op_name)
        self.built = True

    def call(self, inputs):
        if self._is_causal:  # Apply causal padding to inputs for Conv1D.
            inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))

        #norm = normalize_weights(self.kernel)
        #self.kernel = self.normalize_weights(self.kernel)
        #outputs = self._convolution_op(inputs, self.kernel)
        outputs = self._convolution_op(inputs, self.normalize_weights(self.kernel))

        if self.use_bias:
            output_rank = outputs.shape.rank
            if self.rank == 1 and self._channels_first:
                # nn.bias_add does not accept a 1D input tensor.
                bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                outputs += bias
            else:
                # Handle multiple batch dimensions.
                if output_rank is not None and output_rank > 2 + self.rank:

                    def _apply_fn(o):
                        return nn.bias_add(o, self.bias, data_format=self._tf_data_format)

                    outputs = nn_ops.squeeze_batch_dims(
                        outputs, _apply_fn, inner_rank=self.rank + 1)
                else:
                    outputs = nn.bias_add(
                        outputs, self.bias, data_format=self._tf_data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


    def normalize_weights(self, kernel):
        # Spectral normalize, with power iter: 1
        w_shape = kernel.shape.as_list()
        w = tf.reshape(kernel, [-1, w_shape[-1]])

        u_hat = self.u
        v_hat = None

        for _ in range(self.iteration):
            v_ = tf.matmul(u_hat, tf.transpose(w))
            #v_hat = v_ / (tf.reduce_sum(v_**2)**0.5 + self.eps)
            v_hat = tf.math.l2_normalize(v_)

            u_ = tf.matmul(v_hat, w)
            #u_hat = u_ / (tf.reduce_sum(u_**2)**0.5 + self.eps)
            u_hat = tf.math.l2_normalize(u_)

        #u_hat = tf.identity(u_hat)
        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

        #self.u.assign(u_hat)
        #w_norm = kernel / sigma
        with tf.control_dependencies([self.u.assign(u_hat)]):
            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, w_shape)

        return w_norm


    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        batch_rank = len(input_shape) - self.rank - 1
        if self.data_format == 'channels_last':
            return tensor_shape.TensorShape(
                input_shape[:batch_rank]
                + self._spatial_output_shape(input_shape[batch_rank:-1])
                + [self.filters])
        else:
            return tensor_shape.TensorShape(
                input_shape[:batch_rank] + [self.filters] +
                self._spatial_output_shape(input_shape[batch_rank + 1:]))

    def _recreate_conv_op(self, inputs):  # pylint: disable=unused-argument
        return False

    def get_config(self):
        config = {
            'filters':
                self.filters,
            'kernel_size':
                self.kernel_size,
            'strides':
                self.strides,
            'padding':
                self.padding,
            'data_format':
                self.data_format,
            'dilation_rate':
                self.dilation_rate,
            'groups':
                self.groups,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint)
        }
        base_config = super(SNConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _compute_causal_padding(self, inputs):
        """Calculates padding for 'causal' option for 1-d conv layers."""
        left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
        if getattr(inputs.shape, 'ndims', None) is None:
            batch_rank = 1
        else:
            batch_rank = len(inputs.shape) - 2
        if self.data_format == 'channels_last':
            causal_padding = [[0, 0]] * batch_rank + [[left_pad, 0], [0, 0]]
        else:
            causal_padding = [[0, 0]] * batch_rank + [[0, 0], [left_pad, 0]]
        return causal_padding

    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            return -1 - self.rank
        else:
            return -1

    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                        'should be defined. Found `None`.')
        return int(input_shape[channel_axis])

    def _get_padding_op(self):
        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
            op_padding = op_padding.upper()
        return op_padding


class SNConv2D(SNConv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 groups=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(SNConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)
