import tensorflow as tf
import numpy as np
from tensortemplates.conv_res_net import template


def test_conv_res_net() -> None:
    tf.reset_default_graph()
    batch_size = 64
    width = 10
    height = 20
    x = tf.placeholder(tf.float32, shape=(batch_size, width, height))
    y = tf.placeholder(tf.float32, shape=(batch_size, width, height))
    inputs = [x, y]
    shape = (batch_size, height, width, 1)
    input_shapes = [shape, shape]
    output_shapes = [shape, shape]
    kwargs = {'layer_width': 10, 'block_size': 1, 'nblocks': 1, 'width': width,
              'height': height, 'nfilters': 5, 'reuse': False}
    outputs, params = template(inputs, input_shapes, output_shapes, **kwargs)

    # Run the damn thing
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    feed = {t: np.random.rand(*t.get_shape().as_list()) for t in inputs}
    sess.run(outputs, feed_dict=feed)
