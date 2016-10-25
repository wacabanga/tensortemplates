import tensorflow as tf
import numpy as np
from tensortemplates.res_net import template


# def test_res_net() -> None:
#     tf.reset_default_graph()
#     batch_size = 128
#     x = tf.placeholder(tf.float32, shape=(batch_size, 10))
#     y = tf.placeholder(tf.float32, shape=(batch_size, 10, 20))
#     inputs = [x, y]
#     input_shapes = [(batch_size, 10), (batch_size, 10, 20)]
#     output_shapes = [(batch_size, 20), (batch_size, 30)]
#     kwargs = {'layer_width': 10, 'block_size': 1, 'nblocks': 1}
#     outputs, params = template(inputs, input_shapes, output_shapes, **kwargs)
#
#     # Run the damn thing
#     sess = tf.Session()
#     sess.run(tf.initialize_all_variables())
#     feed = {t: np.random.rand(*t.get_shape().as_list()) for t in inputs}
#     sess.run(outputs, feed_dict=feed)


def test_res_net_mnist() -> None:
    tf.reset_default_graph()
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    batch_size = 128
    input_shapes = [(batch_size, 784)]
    output_shapes = [(batch_size, 10)]
    kwargs = {'layer_width': 10, 'block_size': 1, 'nblocks': 1}
    outputs, params = template([x], input_shapes, output_shapes, **kwargs)
    y = outputs[0]

    # Training
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    sess = tf.InteractiveSession()
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    sess.run(tf.initialize_all_variables())
    for i in range(100):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})


    # Evaluation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    score = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    assert score > 0.7, "MNIST score too low"
    sess.close()
    print("Score is", score)

test_res_net_mnist()
