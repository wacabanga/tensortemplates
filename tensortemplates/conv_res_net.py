from typing import Tuple, List, Union
from tensortemplates.util.misc import same
import tensorflow as tf
from tensorflow import Tensor, Variable
from tensorflow.contrib.layers import layer_norm
from tensorflow import nn
import pdb


conv2d = nn.conv2d

TensVar = Union[Tensor, Variable]
ImgShape = Tuple[int, int, int, int]  # (batch, height, width, channels)


def consistent_batch_size(shapes) -> bool:
    """Are the batch sizes the same?"""
    return same([shape[0] for shape in shapes])


def conv_layer(x: TensVar,
               ninp_channels: int,
               nout_channels: int,
               ndim: int,
               sfx: str,
               filter_height=5,
               filter_width=5,
               filter_depth=5,
               nl=tf.nn.elu,
               reuse=False,
               layer_norm=False) -> TensVar:
    """Neural Network Layer - nl(Wx+b)
    x: ImgShape:[batch, in_height, in_width, in_channels]`
    """
    if ndim == 2:
        conv = nn.conv2d
        filter_shape = [filter_height,
                        filter_width,
                        ninp_channels,
                        nout_channels]
        strides = [1, 1, 1, 1]
    else:
        assert ndim == 3
        conv = nn.conv3d
        filter_shape=[filter_height,
                      filter_width,
                      filter_depth,
                      ninp_channels,
                      nout_channels]
        strides = [1, 1, 1, 1, 1]

    with tf.name_scope("conv%sd_layer" % ndim):
        with tf.variable_scope(sfx, reuse=reuse) as scope:
            W = tf.get_variable(name="W_%s" % sfx,
                                shape=filter_shape,
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(name="b_%s" % sfx, shape=[nout_channels],
                                initializer=tf.constant_initializer(0.1))
            outp = conv(x,
                        W,
                        strides=strides,
                        padding='SAME')
            # conv = tf.contrib.layers.batch_norm(conv, is_training=True, trainable=True)
            # conv = tf.contrib.layers.batch_norm(conv, trainable=False, reuse=reuse, scope=scope)
            if layer_norm:
                outp = tf.contrib.layers.layer_norm(outp, reuse=reuse, scope=scope)
            outp = nl(tf.nn.bias_add(outp, b))
    return outp

def residual_block(prev_layer,
                   wx,
                   block_size: int,
                   nprevlayer_channels: int,
                   nlayer_channels: int,
                   ndim: int,
                   do_res_add: bool,
                   reuse: bool,
                   sfx):
    with tf.name_scope("residual_block"):
        for i in range(block_size):
            inner_sfx = "%s_%s" % (sfx, i)
            prev_layer = conv_layer(prev_layer,
                                    ninp_channels=nprevlayer_channels,
                                    nout_channels=nlayer_channels,
                                    ndim=ndim,
                                    sfx=inner_sfx,
                                    reuse=reuse)
            nprevlayer_channels = nlayer_channels
        if do_res_add:
            prev_layer = prev_layer + wx
    return prev_layer


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
            concat_img = tf.concat(input_channels, CHANNEL_DIM)
        return concat_img


def template(inputs: List[TensVar],
             inp_shapes: List[ImgShape],
             out_shapes: List[ImgShape],
             **options) -> Tuple[TensVar, List]:
    """
    2D or 3D convolutional (residual) neural network
    Args:
        inputs : [tf.TensVar/tf.Variable] - inputs to be transformed
        out_shapes : (tf.TensVarImgShape) | (Int) - shapes of output of tensor (includes batch_size)
    Returns:
        outputs: List of tensors
    """
    # Meta Parameters
    nblocks = options['nblocks']
    block_size = options['block_size']
    nfilters = options['nfilters']
    reuse = options['reuse']

    width = options['width']
    height = options['height']

    ndim = max([len(shape) - 2 for shape in inp_shapes])
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

    # Do `nblocks` residual blocks
    for j in range(nblocks):
        prev_layer = residual_block(prev_layer=prev_layer,
                                    wx=wx,
                                    block_size=block_size,
                                    nprevlayer_channels=nprevlayer_channels,
                                    nlayer_channels=nlayer_channels,
                                    ndim=ndim,
                                    do_res_add=nblocks > 1,
                                    reuse=reuse,
                                    sfx=j)
        nprevlayer_channels = nlayer_channels
        wx = prev_layer

    # One final convolutional, can't remember why
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
    options['nl'] = (str, 'elu')
    return options
