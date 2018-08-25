""" Icon (favicon) dectector
This code aims to classify between different types of famus favicons such as:
facebook, google, twitter, etc.

Our assumption is that all the favicons are sharing the size of 16x16 pixels.
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import tempfile
import os
import array
import image_converter

# Data set settings
DATA_SET_PATH = "./input_data/"
NUM_OF_FAVICONS_TYPES = 5
FAVICONS_LABES = [0, 1, 2, 3, 4]
FAVICONS_LABES_NAMES = ["Facebook",
				  "Google",
				  "Outlook",
				  "Twitter",
				  "Paypal"]


# The favicons are always 16x16 pixels
CHANNELS = 1 # Grayscale images
FAVICON_SIZE = 16
FAVICON_PIXELS = FAVICON_SIZE * FAVICON_SIZE * CHANNELS
NUM_OF_FAVICONS = 5

# Graph settings
CONV_SIZE = 5
CONV_FEATURES_1 = 32
CONV_FEATURES_2 = 64
MAP_FEATURE_SIZE = 1024

# Traninig
ALPHA = 1e-4
ROUNDS = 200

def read_data_set(path):
	train_images = []
	train_labels = []
	for favicon_label in FAVICONS_LABES:
		favicon_name = FAVICONS_LABES_NAMES[favicon_label]
		images_path = os.path.join(path, favicon_name)
		RGB_images = [os.path.join(images_path, image) for image in os.listdir(images_path)]
		for image_path in RGB_images:
			image_raw = image_converter.getImageGraysacleArray(image_path, FAVICON_SIZE, FAVICON_SIZE)
			train_images.append(image_raw)
			one_hot = [0] * NUM_OF_FAVICONS
			one_hot[favicon_label] = 1
			train_labels.append(one_hot)

	return (train_images, train_labels)

def cnn_deep_nn(x):
	# Reshape to use within convolution neural network
	with tf.name_scope('reshape'):
		x_image = tf.reshape(x, [-1, FAVICON_SIZE, FAVICON_SIZE, CHANNELS])

	# First convolutional layer - maps one image to #CONV_FEATURES_1 features map
	with tf.name_scope('conv1'):
		W_conv1 = weight_variable([CONV_SIZE, CONV_SIZE, CHANNELS, CONV_FEATURES_1])
		b_conv1 = bias_variable([CONV_FEATURES_1])
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

	# Max pooling layer
	with tf.name_scope('pool1'):
		h_pool1 = max_pool_2x2(h_conv1)

	# Second convolutional layer -- maps #CONV_FEATURES_1 feature maps to #CONV_FEATURES_2.
	with tf.name_scope('conv2'):
		W_conv2 = weight_variable([CONV_SIZE, CONV_SIZE, CONV_FEATURES_1, CONV_FEATURES_2])
		b_conv2 = bias_variable([CONV_FEATURES_2])
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

	# Second max pooling layer
	with tf.name_scope('pool2'):
		h_pool2 = max_pool_2x2(h_conv2)

	# Fully connected layer 1 --- after 2 rounds of downsampling, our FAVICON_SIZE X FAVICON_SIZE image
	# is down to (FAVICON_SIZE / 4) X (FAVICON_SIZE / 4) X #CONV_FEATURES_2 features map - map this to 1024 features
	with tf.name_scope('fc1'):
		current_number_of_features = (FAVICON_SIZE / 4) * (FAVICON_SIZE / 4) * CONV_FEATURES_2
		W_fc1 = weight_variable([current_number_of_features, MAP_FEATURE_SIZE])
		b_fc1 = bias_variable([MAP_FEATURE_SIZE])

		h_pool2_flat = tf.reshape(h_pool2, [-1, current_number_of_features])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	# Dropout
	with tf.name_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	# Map the 1024 features to 5 classes, one for each digit
	with tf.name_scope('fc2'):
		W_fc2 = weight_variable([MAP_FEATURE_SIZE, NUM_OF_FAVICONS_TYPES])
		b_fc2 = bias_variable([NUM_OF_FAVICONS_TYPES])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	return y_conv, keep_prob

def conv2d(x, W):
	"""conv2d returns a 2d convolution layer with full stride."""
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	"""max_pool_2x2 downsamples a feature map by 2X."""
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
						strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
	"""weight_variable generates a weight variable of a given shape."""
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	"""bias_variable generates a bias variable of a given shape."""
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def main():
	(favicons_train_images, favicons_train_labels) = read_data_set(DATA_SET_PATH)

	# Create the model
	x = tf.placeholder(tf.float32, [None, FAVICON_PIXELS])

	# Define loss and optimizer
	y_ = tf.placeholder(tf.float32, [None, NUM_OF_FAVICONS]);

	# Build the graph for the neural network
	y_conv, keep_prob = cnn_deep_nn(x)

	with tf.name_scope('loss'):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
		                                                        logits=y_conv)
		cross_entropy = tf.reduce_mean(cross_entropy)

	with tf.name_scope('adam_optimizer'):
		train_step = tf.train.AdamOptimizer(ALPHA).minimize(cross_entropy)

	with tf.name_scope('accuracy'):
		correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
		correct_prediction = tf.cast(correct_prediction, tf.float32)
		accuracy = tf.reduce_mean(correct_prediction)

	graph_location = tempfile.mkdtemp()
	print 'Saving graph to: %s' % graph_location
	train_writer = tf.summary.FileWriter(graph_location)
	train_writer.add_graph(tf.get_default_graph())

	with tf.Session() as sess:
		try:
			sess.run(tf.global_variables_initializer())
			for i in range(ROUNDS):
				if i % 100 == 0:
					train_accuracy = accuracy.eval(session = sess, feed_dict={
						x: favicons_train_images, y_: favicons_train_labels, keep_prob: 1.0})
					print('step %d, training accuracy %g' % (i, train_accuracy))
				train_step.run(feed_dict={x: favicons_train_images, y_: favicons_train_labels, keep_prob: 0.5})
		except KeyboardInterrupt:
			print "Training aborted by the user"

		print('test accuracy %g' % accuracy.eval(session = sess, feed_dict={
		x: favicons_train_images, y_: favicons_train_labels, keep_prob: 1.0}))


if __name__ == '__main__':
	main()