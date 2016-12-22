'''
Original Author: Aymeric Damien
New Changes/Author: Alisha Rege
'''

from __future__ import print_function

import tensorflow as tf

from random import shuffle
import math
import os.path
import cv2
import numpy as np
import csv
from sklearn.metrics import average_precision_score


# Parameters
learning_rate = 0.01
training_iters = 20000
batch_size = 50
display_step = 10

# Network Parameters
n_input = 226 # data input (img shape: 56*56)
n_classes = 2 # total classes (person/notperson)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [batch_size, 56, 56, 1])
y = tf.placeholder(tf.float32, [batch_size, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Get data - import images
def data_iterator(dataset_path, labels, batch_size, last):
	image_paths = dataset_path[last:last+batch_size]
	images = []
	for image_path in image_paths:
		image = cv2.imread(image_path) # hxw
		if image == None:
			continue
		image = cv2.resize(image, (56, 56))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = image[:, :, np.newaxis] # 56x56x1
		images.append(image)
    
	# images: a length batch_size list of images, each of size 56x56x1
	images = np.stack(images) # batch_sizex56x56x1
	retlabel = labels[last:last+batch_size]
	return (images, retlabel)

# Import my data
yespath = "/afs/cs.stanford.edu/u/amr6114/Labels/Yes"
nopath = "/afs/cs.stanford.edu/u/amr6114/Labels/No"
croppath = "/afs/cs.stanford.edu/u/amr6114/trainingimages"
yes = [os.path.join(croppath,f) for f in os.listdir(yespath) if f.endswith('.jpg')]
no = [os.path.join(croppath,f) for f in os.listdir(nopath) if f.endswith('.jpg')]


#Testing dataset
yespath = "/afs/cs.stanford.edu/u/amr6114/same.10/yes"
croppath = "/afs/cs.stanford.edu/u/amr6114/cropped"
yestest = [os.path.join(croppath,f) for f in os.listdir(yespath) if f.endswith('.jpg')]
yespath = "/afs/cs.stanford.edu/u/amr6114/same.9/yes"
croppath = "/afs/cs.stanford.edu/u/amr6114/cropped2"
yestest = yestest + [os.path.join(croppath,f) for f in os.listdir(yespath) if f.endswith('.jpg')]

nopath = "/afs/cs.stanford.edu/u/amr6114/same.10/No"
croppath = "/afs/cs.stanford.edu/u/amr6114/cropped"
notest = [os.path.join(croppath,f) for f in os.listdir(nopath) if f.endswith('.jpg')]
nopath = "/afs/cs.stanford.edu/u/amr6114/same.9/No"
croppath = "/afs/cs.stanford.edu/u/amr6114/cropped2"
notest = notest + [os.path.join(croppath,f) for f in os.listdir(nopath) if f.endswith('.jpg')]

traindataset = [[f, [1,0]] for f in no] + [[f, [0,1]] for f in yes]
testdataset = [[f, [1,0]] for f in notest] + [[f, [0,1]] for f in yestest]
shuffle(traindataset)
shuffle(traindataset)
shuffle(testdataset)
shuffle(testdataset)

# Let's clean the info aka make the number of persons equal to non people

npe =0
p = 0
for f,l in traindataset:
	if l == [1,0]:
		npe += 1
	if l == [0,1]:
		p += 1
remove = []
for i,(f,l) in enumerate(traindataset):
	if (npe > p):
		if l == [1,0]:
			remove.append(i)
			npe -= 1
	if (p > npe):
		if l == [0,1]:
			remove.append(i)
			p -= 1
for i in reversed(remove):
	traindataset.pop(i)

files = [f for f,l in traindataset]
labels = [l for f,l in traindataset]

testfiles = [f for f,l in testdataset]
testlabels = [l for f,l in testdataset]

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
	# Conv2D wrapper, with bias and relu activation
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)


def maxpool2d(x, k=2):
	# MaxPool2D wrapper
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
	# Reshape input picture
	x = tf.reshape(x, shape=[batch_size, 56, 56, 1])

	# Convolution Layer
	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	# Max Pooling (down-sampling)
	conv1 = maxpool2d(conv1, k=2)

	# Convolution Layer
	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
	# Max Pooling (down-sampling)
	conv2 = maxpool2d(conv2, k=2)
	# Fully connected layer
	# Reshape conv2 output to fit fully connected layer input
	fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)
	# Apply Dropout
	fc1 = tf.nn.dropout(fc1, dropout)
	# Output, class prediction
	out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
	return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([14*14*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

last = 0
numiters = []
for i in range(800):
	numiters.append(40*i)
testiteracc = []
trainiteracc = []
aciter = 0
for training_iters in numiters:
	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		step = 1
		# Keep training until reach max iterations
		i = 0
		while step * batch_size < training_iters:
			i += 1
			if last + batch_size > len(files):
				last = 0
				entiredataset = [[f,l] for f,l in zip(files,labels)]
				shuffle(entiredataset) 
				files = [f for f,l in entiredataset]
				labels = [l for f,l in entiredataset]

			data_set = data_iterator(files, labels, batch_size, last)
			batch_x, batch_y = data_set
			last += batch_size
			# Run optimization op (backprop)
			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
			if step % display_step == 0:
			# Calculate batch loss and accuracy
				loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
				print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  			"{:.6f}".format(loss) + ", Training Accuracy= " + \
                  			"{:.5f}".format(acc))
			step += 1
		print("Optimization Finished!")
    
		# Calculate accuracy for test images
		begin = 0
		labeling = []
		guess = []
		while (begin + batch_size < len(testfiles)):
			data_set = data_iterator(testfiles, testlabels, batch_size, begin)
			begin += batch_size
			y_pred = sess.run(pred, feed_dict={x:data_set[0],
                                      	      y: data_set[1],
                                              keep_prob: 1.})
			if labeling == []:
				labeling = np.argmax(data_set[1],1)
				guess = np.argmax(y_pred,1)
			else:
				labeling = np.append(labeling, np.argmax(data_set[1],1))
				guess = np.append(guess, np.argmax(y_pred,1))
		testiteracc.append([40*aciter, average_precision_score(labeling, guess)])
		begin = 0
		labeling = []
		guess = []
		while (begin + batch_size < len(files)):
			data_set = data_iterator(files, labels, batch_size, begin)
			begin += batch_size
			y_pred = sess.run(pred, feed_dict={x:data_set[0],
                                      	      y: data_set[1],
                                              keep_prob: 1.})
			if labeling == []:
				labeling = np.argmax(data_set[1],1)
				guess = np.argmax(y_pred,1)
			else:
				labeling = np.append(labeling,np.argmax(data_set[1],1))
				guess = np.append(guess,np.argmax(y_pred,1))
		trainiteracc.append([40*aciter, average_precision_score(labeling, guess)])
		aciter += 1
		
np.savetxt('train.csv',trainiteracc, delimiter=',')
np.savetxt('test.csv',testiteracc, delimiter=',')

