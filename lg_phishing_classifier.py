import tensorflow as tf
import pandas as pd
import numpy as np
import sys

PHISHING_DATA = "phishing_sites.csv"
NORMAL_DATA = "normal_sites.csv"

URL_COLUMN_NAME = "url"

FEATURE_URL_LENGTH = "url_length"
FEATURE_DIGITS_COUNT = "digit_count"
FEATURE_DOT_COUNT = "dot_count"
FEATURES = [FEATURE_URL_LENGTH, FEATURE_DIGITS_COUNT, FEATURE_DOT_COUNT]

IS_PHISHING_COLUMN_NAME = "is_phishing"

TRANING_PERCENTAGES = 80
TEST_PERCENTAGES = 20

DATA_SIZE_FROM_EACH_SOURCE = 10000

MAX_STEPS = 10000
ALPHA = 0.00001

THRESHOLD = 0.5

def shuffle(dataFrame):
	return dataFrame.sample(frac=1)

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

def generateClassifier(training_data):
	print "Start training setup"

	(training_rows, training_cols) = training_data.shape
	num_of_features = len(FEATURES)
	epsilon = 1e-12

	x = tf.placeholder(tf.float32, shape = (training_rows, num_of_features))
	y_ = tf.placeholder(tf.float32, shape = (training_rows,))

	W = tf.Variable(tf.zeros([num_of_features, 1]), name = "W")
	b = tf.Variable(tf.zeros([]), name = "b")

	y = 1 / (1.0 + tf.exp(-(tf.matmul(x, W) + b)))
	loss_function = -(y_ * tf.log(y + epsilon) + (1 - y_) * tf.log(1 - y + epsilon))
	loss = tf.reduce_min(loss_function)

	update = tf.train.GradientDescentOptimizer(ALPHA).minimize(loss)

	data_y = training_data[IS_PHISHING_COLUMN_NAME]
	data_x = training_data[FEATURES]

	print "Training setup complete"

	# W_sum = tf.summary.scalar('W', W)
	b_sum = tf.summary.scalar('b', b)
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

	return (session, W, b)

def logistic_regression(t):
	return 1 / (1.0 + np.exp(-t))

def predict(classifier, test):
	(session, W, b) = classifier
	return logistic_regression(np.matmul([test], session.run(W)) + session.run(b))[0][0]

def testData(classifier, test_data):
	(test_rows, test_cols) = test_data.shape

	is_phishing = test_data[IS_PHISHING_COLUMN_NAME].as_matrix()
	test_vectors = test_data[FEATURES].as_matrix()
	urls = test_data[URL_COLUMN_NAME].as_matrix()

	correct_guesses = 0
	for i in range(test_rows):
		prediction = predict(classifier, test_vectors[i]);
		url = urls[i]
		is_phishing_url = is_phishing[i]
		is_phishing_url_prediction = prediction >= THRESHOLD

		if is_phishing_url == is_phishing_url_prediction:
			correct_guesses += 1

		print "test url: ", url, "prediction: ", is_phishing_url_prediction, "reality:", is_phishing_url

	print "summary:"
	print "Tested #" + str(test_rows) + " URLs"
	print "Correct guesses: " + str(correct_guesses) + " (" + str((float(correct_guesses) / test_rows) * 100) + "%)"

def main():
	print "Preparing the data..."
	data = prepareData()
	print "Preparing the data finished"

	(traning_data, test_data) = splitData(data)
	print "Training data size:", traning_data.shape
	print "Test data size:", test_data.shape

	print "Start training"
	(session, W, b) = generateClassifier(traning_data)
	print "Training ended"

	print "Start testing"
	testData((session, W, b), test_data)
	print "Testing ended"

if __name__ == '__main__':
	main()