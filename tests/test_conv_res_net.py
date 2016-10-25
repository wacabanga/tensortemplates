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


def test_conv_res_net_mnist() -> None:
    tf.reset_default_graph()
    from tensorflow.examples.tutorials.mnist import input_data
    batch_size = 50
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    inputs = [x_image]
    input_shapes = [(batch_size, 28, 28, 1)]
    output_shapes = [(batch_size, 28, 28, 1)]
    # output_shapes = [(batch_size, 10)]
    kwargs = {'layer_width': 10, 'block_size': 1, 'nblocks': 1, 'width': 28,
              'height': 28, 'nfilters': 32, 'reuse': False}
    outputs, params = template(inputs, input_shapes, output_shapes, **kwargs)

    y_flat = tf.reshape(outputs[0], [-1, 28 * 28])
    W_dense = tf.Variable(tf.truncated_normal([28 * 28, 10], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[10]))
    y = tf.matmul(y_flat, W_dense) + b

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    sess.run(tf.initialize_all_variables())
    for i in range(100):
        batch = mnist.train.next_batch(batch_size)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        op = sess.run([loss, train_step], feed_dict={x: batch[0], y_: batch[1]})
        print(op)


    # Evaluation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    score = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    assert score > 0.7, "MNIST score of %s too low" % score
    sess.close()
    print("ConvNet Score is", score)

test_conv_res_net_mnist()
