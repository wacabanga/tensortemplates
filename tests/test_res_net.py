import tensorflow as tf
import numpy as np
from tensortemplates.res_net import template


def test_res_net() -> None:
    tf.reset_default_graph()
    batch_size = 128
    x = tf.placeholder(tf.float32, shape=(batch_size, 10))
    y = tf.placeholder(tf.float32, shape=(batch_size, 10, 20))
    inputs = (x, y)
    input_shapes = [(batch_size, 10), (batch_size, 10, 20)]
    output_shapes = [(batch_size, 20), (batch_size, 30)]
    kwargs = {'layer_width': 10, 'block_size': 1, 'nblocks': 1}
    outputs, params = template(inputs, input_shapes, output_shapes, **kwargs)

    # Run the damn thing
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    feed = {t: np.random.rand(*t.get_shape().as_list()) for t in inputs}
    sess.run(outputs, feed_dict=feed)
