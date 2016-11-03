from typing import Tuple, List, Union
from tensortemplates.util.misc import same
import tensorflow as tf
from tensorflow import Tensor, Variable
from tensorflow.contrib.layers import layer_norm
from tensorflow import nn
conv2d = nn.conv2d

TensVar = Union[Tensor, Variable]
ImgShape = Tuple[int, int, int, int]  # (batch, height, width, channels)


def consistent_batch_size(shapes) -> bool:
    """Are the batch sizes the same?"""
    return same([shape[0] for shape in shapes])


def conv_layer(x: TensVar, ninp_channels: int, nout_channels: int, ndim: int,
               sfx: str, filter_height=5, filter_width=5,
               nl=tf.nn.relu, reuse=False) -> TensVar:
    """Neural Network Layer - nl(Wx+b)
    x: ImgShape:[batch, in_height, in_width, in_channels]`
    """
    with tf.name_scope("conv_layer"):
        with tf.variable_scope(sfx) as scope:
            W = tf.get_variable(name="W_%s" % sfx,
                                shape=[filter_height, filter_width,
                                       ninp_channels, nout_channels],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(name="b_%s" % sfx, shape=[nout_channels],
                                initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(x,
                                W,
                                strides=[1, 1, 1, 1],
                                padding='SAME',
                                use_cudnn_on_gpu=True)
            # conv = tf.reduce_mean(conv)
            # conv = tf.contrib.layers.batch_norm(conv, is_training=True, trainable=True)
            # conv = tf.contrib.layers.batch_norm(conv, trainable=False, reuse=reuse, scope=scope)
            op = tf.nn.relu(tf.nn.bias_add(conv, b))
            op = nl(op)
    return op


def unstack_channel(t: TensVar, shapes: List[ImgShape]) -> List[TensVar]:
    """Slice and reshape a flat input (batch_size, n) to list of tensors"""
    assert len(shapes) > 0
    if len(shapes) == 1:
        print("Only one output skipping unstack")
        return [t]
    else:
        with tf.name_scope("unchannel"):
            outputs = []
            for i in range(len(shapes)):
                outputs.append(t[:, :, :, i:i+1])

        return outputs


def stack_channels(inputs: List[TensVar], shapes: List[ImgShape], width: int,
                   height: int, CHANNEL_DIM: int) -> TensVar:
    """Take list of inputs of same size and stack in channel dimension"""
    assert len(inputs) > 0
    if len(inputs) == 1:
        print("Only one input skipping stack")
        return inputs[0]
    else:

        with tf.name_scope("channel"):
            input_channels = []
            for i in range(len(inputs)):
                inp_ndim = len(shapes[i])
                nchannels = 1 # FIXME
                ch = tf.reshape(inputs[i], [tf.shape(inputs[i])[0],
                                height, width, nchannels])
                input_channels.append(ch)
            concat_img = tf.concat(CHANNEL_DIM, input_channels)
        return concat_img


def template(inputs: List[TensVar], inp_shapes: List[ImgShape],
             out_shapes: List[ImgShape], **options) -> Tuple[TensVar, List]:
    """
    Convolutional (residual) neural network
    inputs : [tf.TensVar/tf.Variable] - inputs to be transformed
    out_shapes : (tf.TensVarImgShape) | (Int) - shapes of output of tensor (includes batch_size)
    """
    # Meta Parameters
    nblocks = options['nblocks']
    block_size = options['block_size']
    width, height = options['width'], options['height']
    nfilters = options['nfilters']
    reuse = options['reuse']
    ndim = options['ndim']
    assert ndim == 2 or ndim == 3, "Supports only 2d or 3d convolution"

    if ndim == 2:
        CHANNEL_DIM = 3
    elif ndim == 3:
        CHANNEL_DIM = 4

    assert consistent_batch_size(inp_shapes + out_shapes), "Batch sizes differ"
    prev_layer = stack_channels(inputs, inp_shapes, width, height, CHANNEL_DIM)
    # ninp_channels = tf.shape(prev_layer)[CHANNEL_DIM]

    for shape in out_shapes + inp_shapes:
        if ndim == 2:
            assert len(shape) == 4, "expect batch_size, width, height, nchannels"
        elif ndim == 3:
            assert len(shape) == 5, "expect batch_size, width, height, depth, nchannels"

    ninp_channels = sum([shape[CHANNEL_DIM] for shape in inp_shapes])
    nout_channels = sum([shape[CHANNEL_DIM] for shape in out_shapes])

    # Convolutional Layers
    # ====================
    nprevlayer_channels = ninp_channels
    nlayer_channels = nfilters

    # wx Input Projection
    if nblocks > 1:
        wx = conv_layer(prev_layer,
                        ninp_channels=ninp_channels,
                        nout_channels=nlayer_channels,
                        ndim=ndim,
                        sfx='wx',
                        reuse=reuse)

    # Residual Blocks
    for j in range(nblocks):
        with tf.name_scope("residual_block"):
            for i in range(block_size):
                sfx = "%s_%s" % (j, i)
                prev_layer = conv_layer(prev_layer,
                                        ninp_channels=nprevlayer_channels,
                                        nout_channels=nlayer_channels,
                                        ndim=ndim,
                                        sfx=sfx,
                                        reuse=reuse)
                nprevlayer_channels = nlayer_channels
            if nblocks > 1:
                prev_layer = wx = prev_layer + wx

    prev_layer = conv_layer(prev_layer,
                            ninp_channels=nlayer_channels,
                            nout_channels=nout_channels,
                            ndim=ndim,
                            sfx='final_conv',
                            reuse=reuse)

    # Unconcatenate output and separate
    outputs = unstack_channel(prev_layer, out_shapes)
    params = []  # type: List[Variable]
    return outputs, params


def kwargs():
    """Return (default) arguments for a residual network"""
    options = {}
    options['nblocks'] = (int, 1)
    options['block_size'] = (int, 2)
    options['batch_size'] = (int, 512)
    options['nfilters'] = (int, 24)
    options['ndim'] = (int, 2)
    return options
