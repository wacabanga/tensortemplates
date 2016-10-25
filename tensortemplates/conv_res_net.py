from tensorflow import Tensor
import tensorflow as tf
from tensortemplates.util.misc import same
# from tensorflow.contrib.layers import conv2d
from tensorflow import nn
conv2d = nn.conv2d

CHANNEL_DIM = 3


def consistent_batch_size(shapes) -> bool:
    """Are the batch sizes the same?"""
    return same([shape[0] for shape in shapes])


def conv_layer(x: Tensor, ninp_channels: int, nout_channels: int, sfx: str,
               filter_height=3, filter_width=3,
               nl=tf.nn.relu, reuse=False) -> Tensor:
    """Neural Network Layer - nl(Wx+b)
    x: Shape:[batch, in_height, in_width, in_channels]`
    """
    with tf.name_scope("conv_layer"):
        with tf.variable_scope(sfx) as scope:
            W = tf.get_variable(name="W_%s" % sfx,
                                shape=[filter_height, filter_width,
                                       ninp_channels, nout_channels],
                                initializer=tf.random_uniform_initializer())
            b = tf.get_variable(name="b_%s" % sfx, shape=[nout_channels],
                                initializer=tf.zeros_initializer)
            conv = tf.nn.conv2d(x,
                                W,
                                strides=[1, 1, 1, 1],
                                padding='SAME',
                                use_cudnn_on_gpu=True)
            # conv = tf.reduce_mean(conv)
            # conv = tf.contrib.layers.batch_norm(conv, is_training=True, trainable=True)
            conv = tf.contrib.layers.batch_norm(conv, trainable=False, reuse=reuse, scope=scope)
            op = tf.nn.relu(tf.nn.bias_add(conv, b))
    return op


def unstack_channel(t, shapes):
    """Slice and reshape a flat input (batch_size, n) to list of tensors"""
    with tf.name_scope("unchannel"):
        outputs = []
        for i in range(len(shapes)):
            outputs.append(t[:, :, :, i:i+1])

    return outputs


def stack_channels(inputs, shapes, width: int, height: int) -> Tensor:
    """Take list of inputs of same size and stack in channel dimension"""
    with tf.name_scope("channel"):
        input_channels = []
        for i in range(len(inputs)):
            inp_ndim = len(shapes[i])
            nchannels = 1 # FIXME
            ch = tf.reshape(inputs[i], [tf.shape(inputs[i])[0], height, width, nchannels])
            input_channels.append(ch)
        concat_img = tf.concat(CHANNEL_DIM, input_channels)
    return concat_img


def template(inputs, inp_shapes, out_shapes, **options) -> Tensor:
    """
    Convolutional (residual) neural network
    inputs : [tf.Tensor/tf.Variable] - inputs to be transformed
    out_shapes : (tf.TensorShape) | (Int) - shapes of output of tensor (includes batch_size)
    """
    # Meta Parameters
    nblocks = options['nblocks']
    block_size = options['block_size']
    width, height = options['width'], options['height']
    nfilters = options['nfilters']
    reuse = options['reuse']

    assert consistent_batch_size(inp_shapes + out_shapes), "Batch sizes differ"
    prev_layer = stack_channels(inputs, inp_shapes, width, height)
    # ninp_channels = tf.shape(prev_layer)[CHANNEL_DIM]

    for shape in out_shapes + inp_shapes:
        assert len(shape) == 4, "expect batch_size, width, height, nchannels"

    ninp_channels = sum([shape[CHANNEL_DIM] for shape in inp_shapes])
    nout_channels = sum([shape[CHANNEL_DIM] for shape in out_shapes])

    # Convolutional Layers
    # ====================
    nprevlayer_channels = ninp_channels
    nlayer_channels = nfilters

    # wx Input Projection
    if nblocks > 1:
        wx = conv_layer(prev_layer, ninp_channels=ninp_channels,
                        nout_channels=nlayer_channels, sfx='wx', reuse=reuse)

    # Residual Blocks
    for j in range(nblocks):
        with tf.name_scope("residual_block"):
            for i in range(block_size):
                sfx = "%s_%s" % (j, i)
                prev_layer = conv_layer(prev_layer,
                                        ninp_channels=nprevlayer_channels,
                                        nout_channels=nlayer_channels,
                                        sfx=sfx,
                                        reuse=reuse)
                nprevlayer_channels = nlayer_channels
            if nblocks > 1:
                prev_layer = wx = prev_layer + wx

    prev_layer = conv_layer(prev_layer,
                            ninp_channels=nlayer_channels,
                            nout_channels=nout_channels,
                            sfx='final_conv',
                            reuse=reuse)

    # Unconcatenate output and separate
    outputs = unstack_channel(prev_layer, out_shapes)
    params = []
    return outputs, params


def kwargs():
    """Return (default) arguments for a residual network"""
    options = {}
    options['nblocks'] = (int, 1)
    options['block_size'] = (int, 2)
    options['batch_size'] = (int, 512)
    options['nfilters'] = (int, 24)
    return options
