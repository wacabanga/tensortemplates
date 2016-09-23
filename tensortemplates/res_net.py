import tensorflow as tf
import numpy as np

def same(xs):
    """All elements in xs are the same"""
    if len(xs) == 0:
        return True
    else:
        x1 = xs[0]
        for xn in xs:
            if xn != x1:
                return False

    return True

def consistent_batch_size(shapes):
    """Are the batch sizes the same"""
    return same([shape[0] for shape in shapes])

def batch_flatten(t):
    """Reshape tensor so that it is flat vector (except for batch dimension)"""
    return tf.reshape(t, [tf.shape(t)[0], -1])


def res_net_template_dict(inputs, out_shapes, **kwargs):
    input_list = list(inputs.values())
    out_shapes_list = list(out_shapes.values())
    outputs, params = res_net_template(input_list, out_shapes_list, **kwargs)
    return dict(zip(out_shapes.keys(), outputs)), params

def res_net(inputs, inp_shapes, out_shapes, **kwargs):
    """
    Residual neural network inputs
    inputs : [tf.Tensor/tf.Variable] - inputs to by transformed
    out_shapes : (tf.TensorShape) | (Int) - shapes of output of tensor (includes batch_size)
    """
    ## Meta Parameters
    layer_width = kwargs['layer_width']
    nblocks = kwargs['nblocks']
    block_size = kwargs['block_size']
    output_args = kwargs['output_args']

    ## Handle Reshaping (res_net expects input as vector)
    # inp_shapes = [x.get_shape().as_list() for x in inputs]
    print("Input Shapes to Resnet", inp_shapes)
    print("Output Shapes to Resnet", out_shapes)
    # import pdb; pdb.set_trace()
    assert consistent_batch_size(inp_shapes + out_shapes), "Batch sizes differ"
    flat_input_shapes = [np.prod(inp_shape[1:]) for inp_shape in inp_shapes]
    input_width = np.sum(flat_input_shapes)

    flat_output_shapes = [np.prod(out_shape[1:]) for out_shape in out_shapes]
    output_width = np.sum(flat_output_shapes)

    # Num inputs and outputs
    ninputs = len(inp_shapes)
    noutputs = len(out_shapes)

    params = []

    flat_inputs = [batch_flatten(inp) for inp in inputs]
    x = tf.concat(1, flat_inputs)

    ## Layers
    W = tf.get_variable(name="W", shape=(input_width, output_width), initializer=tf.random_uniform)
    b = tf.get_variable(name="b", shape=(output_width), initializer=tf.zeros)
    l1_op = tf.matmul(x, W) + b
    params = params + [W, b]

    output_product = l1_op
    ## Unconcatenate output and separate
    outputs = []
    lb = 0
    for i in range(noutputs):
        ub = lb + flat_output_shapes[i]
        out = output_product[:, lb:ub]
        new_shape = tf.shape(out)[0] + out_shapes[i][1:]
        rout = tf.reshape(out, new_shape)
        outputs.append(rout)
        lb = ub

    return outputs, params

def res_net_kwargs():
    options = {}
    options['train'] = (True,)
    options['nblocks'] = (int, 1)
    options['block_size'] = (int, 2)
    options['batch_size'] = (int, 512)
    options['nfilters'] = (int, 24)
    options['layer_width'] = (int, 50)
    return options

def test_res_net():
    batch_size = 128
    x = tf.placeholder(tf.float32, shape=(batch_size, 10))
    y = tf.placeholder(tf.float32, shape=(batch_size, 10, 20))
    inputs = (x,y)
    output_shapes = [(batch_size, 20), (batch_size, 30)]
    outputs, params = res_net_template(inputs, output_shapes)

    ## Run the damn thing
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    feed = {t:np.random.rand(*t.get_shape().as_list()) for t in inputs}
    sess.run(outputs, feed_dict=feed)

# test_res_net()
