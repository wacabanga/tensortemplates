import tensorflow as tf
import numpy as np
from tensortemplates.util import *
# from tensorflow.contrib.layers import conv2d
from tensorflow import nn
conv2d = nn.conv2d

CHANNEL_DIM = 3


def consistent_batch_size(shapes):
    """Are the batch sizes the same"""
    return same([shape[0] for shape in shapes])


def conv_layer(x, ninp_channels, nout_channels, sfx, nl=tf.nn.relu, reuse=False):
    """Neural Network Layer - nl(Wx+b)"""
    # import pdb; pdb.set_trace()
    with tf.name_scope("conv_layer"):
        with tf.variable_scope(sfx) as scope:
            W = tf.get_variable(name="W_%s" % sfx,
                                initializer=tf.truncated_normal([5, 5, ninp_channels, nout_channels],
                                                    stddev=0.1,
                                                    dtype=tf.float32))
            b = tf.get_variable(name="b_%s" % sfx,
                                initializer=tf.zeros([nout_channels], dtype=tf.float32))
            conv = tf.nn.conv2d(x,
                                W,
                                strides=[1, 1, 1, 1],
                                padding='SAME')
            op = tf.nn.relu(tf.nn.bias_add(conv, b))
    return op


def unstack_channel(t, shapes):
    """Slice and reshape a flat input (batch_size, n) to list of tensors"""
    with tf.name_scope("unchannel"):
        outputs = []
        for i in range(len(shapes)):
            outputs.append(t[:, :, :, i:i+1])

    # import pdb; pdb.set_trace()
    return outputs


def stack_channels(inputs, shapes, width, height):
    """Take list of inputs of same size and stack in channel dimension"""
    with tf.name_scope("channel"):
        input_channels = []
        for i in range(len(inputs)):
            inp_ndim = len(shapes[i])
            nchannels = 1
            ch = tf.reshape(inputs[i], [tf.shape(inputs[i])[0], height, width, nchannels])
            input_channels.append(ch)
        concat_img = tf.concat(CHANNEL_DIM, input_channels)
    return concat_img


def template(inputs, inp_shapes, out_shapes, **kwargs):
    """
    Residual neural network
    inputs : [tf.Tensor/tf.Variable] - inputs to be transformed
    out_shapes : (tf.TensorShape) | (Int) - shapes of output of tensor (includes batch_size)
    """
    ## Meta Parameters
    nblocks = kwargs['nblocks']
    block_size = kwargs['block_size']
    output_args = kwargs['output_args']
    width, height = kwargs['width'], kwargs['height']
    nfilters = kwargs['nfilters']
    reuse = kwargs['reuse']

    assert consistent_batch_size(inp_shapes + out_shapes), "Batch sizes differ"
    prev_layer = stacked_input = stack_channels(inputs, inp_shapes, width, height)
    # ninp_channels = tf.shape(prev_layer)[CHANNEL_DIM]
    ninp_channels = sum([shape[1] for shape in inp_shapes])
    nout_channels = sum([shape[1] for shape in out_shapes])

    ## Convolutional Layers
    ## ====================if nblocks > 1:

    ## wx Input Projection
    if nblocks > 1:
        wx = conv_layer(prev_layer, ninp_channels=ninp_channels, nout_channels=nout_channels, sfx='wx', reuse=reuse)

    nprevlayer_channels = ninp_channels
    nlayer_channels = nfilters
    ## Residual Blocks
    for j in range(nblocks):
        with tf.name_scope("residual_block"):
            for i in range(block_size):
                sfx = "%s_%s" % (j, i)
                prev_layer = output = conv_layer(prev_layer,
                                                 ninp_channels=nprevlayer_channels,
                                                 nout_channels=nlayer_channels,
                                                 sfx=sfx,
                                                 reuse=reuse)
                nprevlayer_channels = nlayer_channels
            if nblocks > 1:
                prev_layer = wx = prev_layer + wx

    # import pdb; pdb.set_trace()
    prev_layer = conv_layer(prev_layer,
                            ninp_channels=nlayer_channels,
                            nout_channels=nout_channels,
                            sfx='final_conv',
                            reuse=reuse)
    nout_channels
    ## Unconcatenate output and separate
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
