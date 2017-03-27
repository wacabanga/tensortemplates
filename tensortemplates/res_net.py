import tensorflow as tf
from tensorflow import Tensor
import numpy as np
from tensortemplates.util.misc import same


def consistent_batch_size(shapes) -> bool:
    """Are the batch sizes the same"""
    return same([shape[0] for shape in shapes])


def batch_flatten(t: Tensor) -> Tensor:
    """Reshape tensor so that it is flat vector (except for batch dimension)"""
    return tf.reshape(t, [tf.shape(t)[0], -1])


def template_dict(inputs, inp_shapes, out_shapes, **kwargs):
    input_list = list(inputs.values())
    inp_shapes_list = list(inp_shapes.values())
    out_shapes_list = list(out_shapes.values())
    outputs, params = template(input_list, inp_shapes_list, out_shapes_list, **kwargs)
    return dict(zip(out_shapes.keys(), outputs)), params


def layer(x: Tensor, inp_width: int, out_width: int, sfx: str, nl=tf.nn.elu,
          W_init=None, b_init=None, batch_norm: bool=True, reuse=False):
    """Neural Network Layer - nl(Wx+b)"""
    # import pdb; pdb.set_trace()
    assert tf.get_variable_scope().reuse == reuse
    with tf.name_scope("layer"):
        with tf.variable_scope(sfx, reuse=reuse) as scope:
            assert tf.get_variable_scope().reuse == reuse
            # import pdb; pdb.set_trace()
            W = tf.get_variable(name="W_%s" % sfx, shape=(inp_width, out_width),
                                initializer=W_init)
            b = tf.get_variable(name="b_%s" % sfx, shape=(out_width,),
                                initializer=b_init)
            mmbias = tf.matmul(x, W) + b
            if batch_norm:
                mmbias = tf.contrib.layers.batch_norm(mmbias, reuse=reuse, scope=scope, is_training=False)
            op = nl(mmbias, name='op_%s' % sfx)
    return op


def flat_shape(shape):
    """Return the flattened shape of a tensor"""
    return np.prod(shape[1:])


def width(shapes):
    flat_shapes = [flat_shape(shape) for shape in shapes]
    return np.sum(flat_shapes)


def sliceup(t, shapes):
    """Slice and reshape a flat input (batch_size, n) to list of tensors"""
    with tf.name_scope("sliceup"):
        noutputs = len(shapes)
        outputs = []
        lb = 0
        flat_shapes = [flat_shape(shape) for shape in shapes]
        for i in range(noutputs):
            ub = lb + flat_shapes[i]
            out = t[:, lb:ub]
            # import pdb; pdb.set_trace()
            new_shape = (tf.shape(out)[0],) + shapes[i][1:]
            rout = tf.reshape(out, new_shape)
            outputs.append(rout)
            lb = ub
    return outputs


def template(inputs, inp_shapes, out_shapes, **kwargs):
    """
    Residual neural network
    Args:
        inputs : [tf.Tensor/tf.Variable] - inputs to be transformed
        out_shapes : (tf.TensorShape) | (Int) - shapes of output of tensor (includes batch_size)
        batch_norm: Apply batch normalization to layers
        skip_last_nl: Skip nonlinearity on last layer
    """
    # Meta Parameters
    layer_width = kwargs['layer_width']
    nblocks = kwargs['nblocks']
    block_size = kwargs['block_size']
    reuse = kwargs['reuse']
    batch_norm = kwargs['batch_norm']
    nl = kwargs['nl']

    # Handle Reshaping (res_net expects input as vector)
    print("Input Shapes to Resnet", inp_shapes)
    print("Output Shapes to Resnet", out_shapes)
    assert consistent_batch_size(inp_shapes + out_shapes), "Batch sizes differ"
    input_width = width(inp_shapes)
    output_width = width(out_shapes)

    # Num inputs and outputs
    flat_inputs = [batch_flatten(inp) for inp in inputs]
    prev_layer = tf.concat(flat_inputs, 1)

    ## Layers
    ## ======

    ## Input Projection
    if nblocks > 1:
        if layer_width != input_width:
            print("Input projection, layer_width: %s input_width: %s" % (layer_width, input_width))
            wx_sfx = 'wxinpproj'
            wx = layer(prev_layer, input_width, layer_width, wx_sfx,
                       batch_norm=batch_norm, reuse=reuse, nl=nl)
        else:
            print("Skipping input weight projection, layer_width: %s input_width: %s" % (layer_width, input_width))
            wx = prev_layer

    # On first layer only prev_layer_width = input_width
    prev_layer_width = input_width
    # prev_layer = tf.Print(prev_layer, [prev_layer], message="resnetinp")
    ## Residual Blocks
    for j in range(nblocks):
        with tf.name_scope("residual_block"):
            for i in range(block_size):
                print("IJ",i," ", j)
                sfx = "%s_%s" % (j, i)
                prev_layer = output = layer(prev_layer, prev_layer_width,
                                            layer_width, sfx,
                                            batch_norm=batch_norm,
                                            reuse=reuse,
                                            nl=nl)

                 # On all other layers prev_layer_width = layer_width
                prev_layer_width = layer_width
            if nblocks > 1:
                prev_layer = wx = prev_layer + wx

    ## Project output to correct width
    if layer_width != output_width:
        print("Output projection, layer_width: %s output_width: %s" % (layer_width, output_width))
        wx_sfx = 'wxoutproj'
        prev_layer = layer(prev_layer, layer_width, output_width, wx_sfx,
                           batch_norm=batch_norm, reuse=reuse, nl=nl)
    else:
        print("Skipping output projection, layer_width: %s output_width: %s" % (layer_width, output_width))

    with tf.name_scope("bias_add"):
        b = tf.get_variable(name="final_bias", shape=(output_width),
                            initializer=None)
        prev_layer = prev_layer + b
    ## Unconcatenate output and separate
    outputs = sliceup(prev_layer, out_shapes)
    params = []
    return outputs, params


def kwargs():
    """Return (default) arguments for a residual network"""
    options = {}
    options['nblocks'] = (int, 1)
    options['block_size'] = (int, 1)
    options['layer_width'] = (int, 50)
    options['batch_norm'] = (bool, True)
    options['nl'] = (str, 'elu')
    return options
