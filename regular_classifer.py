import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from operator import itemgetter
from math import log
from tensorflow.examples.tutorials.mnist import input_data
import functools
import os

# ((train_data, train_labels),
#  (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()
# train_data = train_data/np.float32(255)
# eval_data = eval_data/np.float32(255)

# train_X = train_data.reshape(-1, 28, 28, 1)
# test_X = eval_data.reshape(-1, 28, 28, 1)
# train_y = train_labels
# test_y = eval_labels
next_percent_train = 0.18
selection_threshold = 0.005
decay_rate = 0.00033
length_most_conf = 0
train_size = 55000
cur_percent_train = 0.1
increment_rate = 0.05
training_iters = 9
class prob_element(object): 
	def __init__(self, pos, prob_vec):
		self.pos_ = pos
		self.prob_vec_ = prob_vec 

# def prob_element_sort(a, b):
# 	if (a.prob_vec_)
data = input_data.read_data_sets('MNIST_data',one_hot=True)
train_X = data.train.images.reshape(-1, 28, 28, 1)
test_X = data.test.images.reshape(-1, 28, 28, 1)
train_y = data.train.labels
test_y = data.test.labels


print(train_X.shape, " This is the initial train shape")
print(train_y.shape, " This is the initial train labels")
print(test_X.shape, " This is the initial test shape")
print(test_y.shape, " This is the initial test labels")
training_iterations = 200
learning_rate = 0.001
batch_size = 128

n_input = 28 
n_classes = 10 

x = tf.placeholder("float", [None, 28, 28, 1])
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

with tf.Session() as sess:
    sess.run(init) 
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    index = int(cur_percent_train*train_size)
    temp_train_X = train_X[index:]
    train_X = train_X[:index]
    temp_train_y = train_y[index:]
    train_y = train_y[:index]

    for i in range(training_iters):
        for batch in range(len(train_X)//batch_size):
            batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
            batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]    
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x,
                                                              y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
        print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")

        # Calculate accuracy for all 10000 mnist test images
        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X,y : test_y})
        index = int(increment_rate*train_size)
        train_X = np.concatenate((train_X, temp_train_X[:index]), axis=0)
        train_y = np.concatenate((train_y, temp_train_y[:index]), axis=0)
        temp_train_X = temp_train_X[index:]
        temp_train_y = temp_train_y[index:]
        cur_percent_train = cur_percent_train + increment_rate
        print(cur_percent_train)
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:","{:.5f}".format(test_acc))
    summary_writer.close()