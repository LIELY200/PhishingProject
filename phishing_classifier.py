import tensorflow as tf
import pandas as pd
import numpy as np
import sys

# Data settings
PHISHING_DATA = "phishing_sites.csv"
NORMAL_DATA = "normal_sites.csv"

# CSV data structe
URL_COLUMN_NAME = "url"
IS_PHISHING_COLUMN_NAME = "is_phishing"

# Features
FEATURE_URL_LENGTH = "url_length"
FEATURE_DIGITS_COUNT = "digit_count"
FEATURE_DOT_COUNT = "dot_count"
FEATURES = [FEATURE_URL_LENGTH, FEATURE_DIGITS_COUNT, FEATURE_DOT_COUNT]


# Traninig sizes
DATA_SIZE_FROM_EACH_SOURCE = 4
TEST_PERCENTAGES = 20
TRANING_PERCENTAGES = 100 - TEST_PERCENTAGES

# Traning settings
EPSILON = 1e-12
MAX_STEPS = 10000
ALPHA = 0.001
DEVIATION = 0.1
BIAS_INIT_NUM = 0.1

# Classifier settings
HIDDEN_LAYERS_COUNT = 3
THRESHOLD = 0.4

def shuffle(dataFrame):
	return dataFrame.sample(frac = 1)

def countDigitsInURL(url):
	count = 0
	for character in url:
		if character >= '0' and character <= '9':
			count += 1
	return count

def countDotsInURL(url):
	count = 0
	for character in url:
		if character == '.':
			count += 1
	return count

def addDotsCount(site_data):
	site_data[FEATURE_DOT_COUNT] = site_data[URL_COLUMN_NAME].map(lambda x: countDotsInURL(x))

def addDigitCount(site_data):
	site_data[FEATURE_DIGITS_COUNT] = site_data[URL_COLUMN_NAME].map(lambda x: countDigitsInURL(x))	

def addURLLength(site_data):
	site_data[FEATURE_URL_LENGTH] = site_data[URL_COLUMN_NAME].map(lambda x: len(x))

def addFeatures(site_data):
	addURLLength(site_data)
	addDotsCount(site_data)
	addDigitCount(site_data)

def prepareData():
	phishing_sites = pd.read_csv(PHISHING_DATA, quotechar='"')
	normal_sites = pd.read_csv(NORMAL_DATA, quotechar='"')

	phishing_sites = phishing_sites[:DATA_SIZE_FROM_EACH_SOURCE]
	addFeatures(phishing_sites)
	phishing_sites[IS_PHISHING_COLUMN_NAME] = True

	normal_sites = normal_sites[:DATA_SIZE_FROM_EACH_SOURCE]
	addFeatures(normal_sites)
	normal_sites[IS_PHISHING_COLUMN_NAME] = False

	return shuffle(pd.concat([phishing_sites, normal_sites]))

def splitData(data):
	(num_rows, num_cols) = data.shape
	training_sample_size = int(num_rows * (TRANING_PERCENTAGES / 100.0))
	
	training_data = data[:training_sample_size]
	test_data = data[training_sample_size:]

	return (training_data, test_data)

def generateNeuralNetworkClassifier(training_data):
	print "Start training setup"

	(training_rows, training_cols) = training_data.shape
	num_of_features = len(FEATURES)

	x = tf.placeholder(tf.float32, shape = (training_rows, num_of_features))
	y_ = tf.placeholder(tf.float32, shape = (training_rows,))

	W1 = tf.Variable(tf.truncated_normal([num_of_features, HIDDEN_LAYERS_COUNT], stddev = DEVIATION), name = "W_matrix")
	b1 = tf.Variable(tf.constant(BIAS_INIT_NUM, shape = [HIDDEN_LAYERS_COUNT]), name = "b_vector")
	z1 = tf.add(tf.matmul(x, W1), b1)

	a1 = tf.nn.relu(z1)
	W2 = tf.Variable(tf.truncated_normal([HIDDEN_LAYERS_COUNT, 1], stddev = DEVIATION), name = "W")
	b2 = tf.Variable(0., name = "b")
	z2 = tf.matmul(a1, W2) + b2

	y = 1 / (1.0 + tf.exp(-z2))
	loss_function = -(y_ * tf.log(y + EPSILON) + (1 - y_) * tf.log(1 - y + EPSILON))
	loss = tf.reduce_min(loss_function)

	update = tf.train.GradientDescentOptimizer(ALPHA).minimize(loss)

	data_y = training_data[IS_PHISHING_COLUMN_NAME]
	data_x = training_data[FEATURES]

	print "Training setup complete"

	# Adding graphs for "bias" and "loss"
	b_sum = tf.summary.scalar('b', b2)
	loss_sum = tf.summary.scalar('loss', loss)
	merged = tf.summary.merge_all()

	session = tf.Session()

	session.run(tf.global_variables_initializer())
	file_writer = tf.summary.FileWriter("./graphs", session.graph)

	print "Training on the data strated"

	process_percentages = 0;
	last_printed_process_percentages = -1;
	try:
		for i in xrange(MAX_STEPS):
			_, current_summary = session.run([update, merged], feed_dict = {x: data_x, y_: data_y})
			file_writer.add_summary(current_summary, i)

			process_percentages = int((float(i) / MAX_STEPS) * 100);
			if process_percentages != last_printed_process_percentages:
				last_printed_process_percentages = process_percentages;
				print "%d%%" % (process_percentages)
	except KeyboardInterrupt:
		print "Loss function minimization interrupted"
	finally:
		file_writer.close()	

	print "Training on the data completed successfully"

	return (session, W1, b1, W2, b2)

def sigmoid(t):
	return 1 / (1.0 + np.exp(-t))

def predict(classifier, test):
	(session, W1, b1, W2, b2) = classifier
	before_relu = np.matmul([test], session.run(W1)) + session.run(b1)
	return sigmoid(np.matmul(before_relu, session.run(W2)) + session.run(b2))[0][0]

def testData(classifier, training_data, test_data):
	(validation_test_rows, _) = test_data.shape
	(training_test_rows, _) = training_data.shape 

	print "Training tests:"
	(training_correct_guesses, training_mistake) = testDataInternal(classifier, training_data)

	print "Validation tests:"
	(validation_correct_guesses, validation_mistake) = testDataInternal(classifier, test_data)

	print "Summary:"
	print "Validation Tests"
	print "\tTested #" + str(validation_test_rows) + " URLs"
	print "\tCorrect guesses: " + str(validation_correct_guesses) + " (" + str((float(validation_correct_guesses) / validation_test_rows) * 100) + "%)"
	print "\tAggregated mistake from threshold " + str(validation_mistake)
	print "\tAverage mistake from threshold " + str(float(validation_mistake) / validation_test_rows)
	print "Training Tests"
	print "\tTested #" + str(training_test_rows) + " URLs"
	print "\tCorrect guesses: " + str(training_correct_guesses) + " (" + str((float(training_correct_guesses) / training_test_rows) * 100) + "%)"
	print "\tAggregated mistake from threshold " + str(training_mistake)
	print "\tAverage mistake from threshold " + str(float(training_mistake) / training_test_rows)

def testDataInternal(classifier, data):
	(data_size, _) = data.shape

	is_phishing = data[IS_PHISHING_COLUMN_NAME].as_matrix()
	test_vectors = data[FEATURES].as_matrix()
	urls = data[URL_COLUMN_NAME].as_matrix()

	aggregated_threshold_distance_when_wrong = 0

	correct_guesses = 0
	for i in range(data_size):
		prediction = predict(classifier, test_vectors[i]);
		url = urls[i]
		is_phishing_url = is_phishing[i]
		is_phishing_url_prediction = prediction >= THRESHOLD

		if is_phishing_url == is_phishing_url_prediction:
			correct_guesses += 1
		else:
			aggregated_threshold_distance_when_wrong += abs(prediction - THRESHOLD)

		print "test url: ", url, "prediction: ", is_phishing_url_prediction, "reality:", is_phishing_url

	return (correct_guesses, aggregated_threshold_distance_when_wrong)

def main():
	print "Preparing the data..."
	data = prepareData()
	print "Preparing the data finished"

	(training_data, test_data) = splitData(data)
	print "Training data size:", training_data.shape
	print "Test data size:", test_data.shape

	print "Start training"
	(session, W1, b1, W2, b2) = generateNeuralNetworkClassifier(training_data)
	print "Training ended"

	print "Start testing"
	testData((session, W1, b1, W2, b2), training_data, test_data)
	print "Testing ended"

if __name__ == '__main__':
	main()