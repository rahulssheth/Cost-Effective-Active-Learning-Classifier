import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from operator import itemgetter
from math import log
from tensorflow.examples.tutorials.mnist import input_data
import functools
import os

next_percent_train = 0.18
selection_threshold = 0.005
decay_rate = 0.00033
length_most_conf = 0
train_size = 55000

class prob_element(object): 
	def __init__(self, pos, prob_vec):
		self.pos_ = pos
		self.prob_vec_ = prob_vec 

data = input_data.read_data_sets('MNIST_data',one_hot=True)
train_X = data.train.images.reshape(-1, 28, 28, 1)
test_X = data.test.images.reshape(-1, 28, 28, 1)
train_y = data.train.labels
test_y = data.test.labels

learning_rate = 0.001
batch_size = 128

n_input = 28 
n_classes = 10 

x = tf.placeholder("float", [None, n_input, n_input, 1])
y = tf.placeholder("float", [None, n_classes])

def conv2d(x, W, b, strides=1):
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)

def maxpool2d(x, k=2):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

weights = {
	'wc1': tf.get_variable('W0', shape=(3, 3, 1, 32), initializer=tf.contrib.layers.xavier_initializer()),
	'wc2': tf.get_variable('W1', shape=(3, 3, 32, 64), initializer=tf.contrib.layers.xavier_initializer()),
	'wc3': tf.get_variable('W2', shape=(3, 3, 64, 128), initializer=tf.contrib.layers.xavier_initializer()),
	'wd1': tf.get_variable('W3', shape=(4*4*128, 128), initializer=tf.contrib.layers.xavier_initializer()),
	'out': tf.get_variable('W6', shape=(128, n_classes), initializer=tf.contrib.layers.xavier_initializer()), 
}

biases = {
	'bc1': tf.get_variable('b1', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
	'bc2': tf.get_variable('b2', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
	'bc3': tf.get_variable('b3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
	'bc4': tf.get_variable('b4', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
	'bc5': tf.get_variable('b5', shape=(n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}

def conv_net(x, weights, biases):
	conv1 = conv2d(x, weights['wc1'], biases['bc1'])

	conv1 = maxpool2d(conv1, k=2)

	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])

	conv2 = maxpool2d(conv2, k=2)

	conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
	conv3 = maxpool2d(conv3, k=2)

	fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bc4'])
	fc1 = tf.nn.relu(fc1)

	out = tf.add(tf.matmul(fc1, weights['out']), biases['bc5'])

	return out

def compare(item1, item2):
	if item1.prob_vec_ > item2.prob_vec_:
		return True
	else:
		return False 
# Returns a list of indices

pred = conv_net(x, weights, biases)
probabilities = tf.nn.softmax(pred)
classes = tf.argmax(input=probabilities, axis=1)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()


# Have to run argmax on most_conf_set and then pass those samples over to the training set. 

# With least_conf_set, I would extract all of the y_values from the test_set. 
# Then for the high_conf_set, I would run argmax to extract all of them. But I would still need to 

# Both give me a list of indices
def calc_entropy(prob_list):
	sum = 0
	for each in prob_list:
		entropy_val = -1*each * log(each)
		sum = sum + entropy_val
	return sum  

prev_high_conf = []

with tf.Session() as sess:
	sess.run(init)
	train_loss = []
	test_loss = []
	train_accuracy = []
	test_accuracy = []
	new_train_size = int(.1*train_size)
	valid_X = train_X[new_train_size:]
	valid_y = train_y[new_train_size:]
	train_X = train_X[:new_train_size]
	train_y = train_y[:new_train_size]
	for batch in range(len(train_X)//batch_size):
		batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
		batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]   
        # Run optimization op (backprop).
        # Calculate batch loss and accuracy
		opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
		loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})

	print("Iter " + str(0) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
    # Calculate accuracy for all 10000 mnist test images
	test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X,y : test_y})
	class_ = sess.run([classes], feed_dict={x: test_X, y: test_y})
	prob = sess.run([probabilities], feed_dict={x: valid_X})
	least_conf = []
	most_conf = []
	elem_list = []
	for each in prob:
		for each_ind in range(0, len(each)):
			if (calc_entropy(each[each_ind]) < selection_threshold):
				most_conf.append(each_ind)
			prob_element_val = prob_element(each_ind, max(each[each_ind]))
			elem_list.append(prob_element_val)

	elem_list = sorted(elem_list, key=lambda prob_elem: prob_elem.prob_vec_)
	elem_list = elem_list[:1500]
	for each in elem_list:
		least_conf.append(each.pos_)
	temp_most_conf = []
	for each in most_conf:
		if each not in least_conf:
			temp_most_conf.append(each)
	most_conf = temp_most_conf
	len_necessary_train = next_percent_train * train_size
	len_necessary_most_conf = int(len_necessary_train - len(least_conf) - len(train_X))

	if len_necessary_most_conf < len(most_conf):
		most_conf = most_conf[:len_necessary_most_conf]
	pseudo_labels = []
	for each in most_conf:
		cur_prob = prob[0][each]
		max_pos = np.argmax(cur_prob)
		one_hot_arr = [0 for _ in range(10)]
		one_hot_arr[max_pos] = 1
		pseudo_labels.append(one_hot_arr)

	next_percent_train = next_percent_train + .1
	# update_train_sets(pseudo_labels, train_X, train_y, valid_X, valid_y, least_conf, most_conf)
	for each in least_conf:
		train_X = np.append(train_X, [valid_X[each]], axis=0)
		train_y = np.append(train_y, [valid_y[each]], axis=0)
	for each in range(0, len(most_conf)):
		train_X = np.append(train_X, [valid_X[most_conf[each]]], axis=0)
		train_y = np.append(train_y, [pseudo_labels[each]], axis=0)
	full_delete = list(set(least_conf + most_conf))	
	full_delete = sorted(full_delete, reverse=True)
	length_most_conf = len(most_conf)
	for each in full_delete:
		if each < len(valid_X):
			valid_X = np.delete(valid_X, each, axis=0)
			valid_y = np.delete(valid_y, each, axis=0)
	train_loss.append(loss)
	test_loss.append(valid_loss)
	train_accuracy.append(acc)
	test_accuracy.append(test_acc)
	print("Initial Testing Accuracy:","{:.5f}".format(test_acc))
	print(len(train_X) / train_size, " This is the amount of labelled training data")
	iter = 0
	while iter < 20 and len(valid_X) > 0:
		# First train:
		for batch in range(len(train_X)//batch_size):
			batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
			batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]   
	        # Run optimization op (backprop).
	        # Calculate batch loss and accuracy
			opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
			loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
	                                                              y: batch_y})
		# Then extract the old most confident set from the train set. Delete from train but do not add to test

		print("Iter " + str(0) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
		rev_train_set_X = np.flip(train_X)
		rev_train_set_y = np.flip(train_y)
		test_set_X = rev_train_set_X.copy()
		test_set_y = rev_train_set_y.copy()
		test_set_X = test_set_X[:length_most_conf]
		test_set_y = test_set_y[:length_most_conf]
		test_set_X = np.flip(test_set_X)
		test_set_y = np.flip(test_set_y)
		train_X = train_X[:len(train_X) - length_most_conf]
		train_y = train_y[:len(train_y) - length_most_conf]
		# Append the old most confident set over to the test set again. 
		test_set_X = test_set_X.reshape(-1, n_input, n_input, 1)
		valid_X = np.concatenate((valid_X, test_set_X), axis=0)
		valid_y = np.concatenate((valid_y, test_set_y), axis=0)

		prob = sess.run([probabilities], feed_dict={x: valid_X})
		test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X,y : test_y})
		least_conf = []
		most_conf = []
		selection_threshold = selection_threshold - decay_rate
		elem_list = []
		for each in prob:
			for each_ind in range(0, len(each)):
				if (calc_entropy(each[each_ind]) < selection_threshold):
					most_conf.append(each_ind)
				prob_element_val = prob_element(each_ind, max(each[each_ind]))
				elem_list.append(prob_element_val)

		elem_list = sorted(elem_list, key=lambda prob_elem: prob_elem.prob_vec_)
		elem_list = elem_list[:1500]
		for each in elem_list:
			least_conf.append(each.pos_)
		temp_most_conf = []
		for each in most_conf:
			if each not in least_conf:
				temp_most_conf.append(each)
		most_conf = temp_most_conf
		len_necessary_train = next_percent_train * train_size
		len_necessary_most_conf = int(len_necessary_train - len(least_conf) - len(train_X))

		if len_necessary_most_conf < len(most_conf):
			most_conf = most_conf[:len_necessary_most_conf]
		pseudo_labels = []
		for each in most_conf:
			cur_prob = prob[0][each]
			max_pos = np.argmax(cur_prob)
			one_hot_arr = [0 for _ in range(10)]
			one_hot_arr[max_pos] = 1
			pseudo_labels.append(one_hot_arr)
		next_percent_train = next_percent_train + .1
		# Run the test set. Extract the new least and most confident sets. Add them all into the train set. 
		for each in least_conf:
			train_X = np.append(train_X, [valid_X[each]], axis=0)
			train_y = np.append(train_y, [valid_y[each]], axis=0)
		for each in range(0, len(most_conf)):
			train_X = np.append(train_X, [valid_X[most_conf[each]]], axis=0)
			train_y = np.append(train_y, [pseudo_labels[each]], axis=0)
		full_delete = list(set(least_conf + most_conf))	
		full_delete = sorted(full_delete, reverse=True)
		length_most_conf = len(most_conf)
		for each in full_delete:
			if each < len(valid_X):
				valid_X = np.delete(valid_X, each, axis=0)
				valid_y = np.delete(valid_y, each, axis=0)	
		train_loss.append(loss)
		test_loss.append(valid_loss)
		train_accuracy.append(acc)
		test_accuracy.append(test_acc)
		print("New Testing Accuracy:","{:.5f}".format(test_acc))
		print(len(train_X) / train_size, " This is the amount of labelled training data")
		iter = iter + 1