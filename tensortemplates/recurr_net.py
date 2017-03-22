import tensorflow as tf 
import numpy as np
from tensorflow.contrib import rnn

'''
input shapes, output shapes
-how to split up the input data (batch_size, 32, 32, 32) -> whatever I want
'''
def consistent_batch_size(shapes) -> bool:
    """Are the batch sizes the same?"""
    return same([shape[0] for shape in shapes])


def rnn_layer(x, n_hidden, n_classes):

	weights = {'out':tf.Variable(tf.random_normal([n_hidden, n_classes]))}
	biases = {'out': tf. Variable(tf.random_normal([n_classes]))}
	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, n_input])
	x = tf.split(x, n_steps,0)
	lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
	outputs, states = rnn.static_rnn(ltsm_cell, x, dtype=tf.float32)
	return tf.matmul(outputs[-1], weights['out'], biases['out'])


def rnn_template(inputs, 
				inp_shapes, 
				out_shapes,

	# Parameters
	learning_rate = 0.001
	training_iters = 100000
	batch_size = 10
	display_step = 10

	# Network Parameters
	n_input = 10
	n_steps = 10  	# timesteps
	n_hidden = 20 	# hidden layer number of features 
	n_classes = 10	# MNIST classes; change

	assert consistent_batch_size(inp_shapes + out_shapes), "Batch sizes differ"

	lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

	inp = tf.placeholder('float', [None, n_steps, n_input])
	out = tf.placeholder('float', [None, n_classes])

	pred = rnn_layer(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=out_shapes))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		step = 1
		while step * batch_size < training_iters:
			batch_x, batch_y = next_batch(batch_size) # next data point; TODO
			batch_x = batch_x.reshape((batch_size, n_steps, n_input))
			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

			if step % display_step == 0:
				acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
				loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
				print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
					"{:.6f}".format(loss) + ", Training Accuracy= " + \
					"{:.5f}".format(acc))

			step+=1
		print("Optimization Finished!")
	return outputs, params
		# DO testing?
