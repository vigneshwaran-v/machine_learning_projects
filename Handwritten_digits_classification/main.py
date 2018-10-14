# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:14:29 2017

@author: Vigneshwaran Vasanthakumar
"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import numpy as np
import os
from PIL import Image
import glob
import tensorflow as tf

#Student Information
print('UBitName = vvasanth')
print('personNumber = 50248708')
print('UBitName = ss623')
print('personNumber = 50247317')
print('UBitName = rajvinod')
print('personNumber = 50247214')
#initializing the image and label placeholders for USPS data
X_images = []
X_labels = []

#Formatting USPS data to match the MNIST data format
for i in range(0, 10):
    total = len(glob.glob1("proj3_images\\USPS\\" + str(i), "*.png"))
    j = 0
    for filename in os.listdir("proj3_images\\USPS\\" + str(i)):
        if filename.endswith('.png'):
            pic = Image.open("proj3_images\\USPS\\" + str(i) + "\\" + filename).convert('L')

            pic_w, pic_h = pic.size
            if(pic_h > pic_w):
                background = Image.new('L', (pic_h, pic_h), color=255)
            else:
                background = Image.new('L', (pic_w, pic_w), color=255)

            bg_w, bg_h = background.size
            offset = (int((bg_w - pic_w) / 2), int((bg_h - pic_h) / 2))
            background.paste(pic, offset)

            pic = background

            pic = pic.resize((28, 28), Image.ANTIALIAS)
            pic = np.array(pic)
            pic = pic/255.0

            pic = np.sin(np.arccos(pic))

            super_threshold_indices = pic < 0.2
            pic[super_threshold_indices] = 0

            pic = pic.reshape(784)
            ans = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            X_images.append(pic)
            ans[i] = 1
            X_labels.append(ans)

#Declaring placeholders for input,output,weight and bias
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

#calculating the cost function & updating the optimized weights
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step =tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
#training
for _ in range(12000):
    batch_xs, batch_ys = mnist.train.next_batch(300)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy =tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print('Logistic Regression train accuracy:' + str(sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.train.labels})))
print('Logistic Regression test accuracy(MNIST):' + str(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))
print('Logistic Regression test accuracy(USPS):' + str(sess.run(accuracy, feed_dict={x: X_images, y_: X_labels})))

#Multilayer perceptron with one hidden layer

#Declaring placeholders for input,hideen layer,output,weights and bias
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W1 = tf.Variable(tf.random_normal([784, 250], stddev=0.04))
b1 = tf.Variable(tf.random_normal([250]))
W2 = tf.Variable(tf.random_normal([250, 10], stddev=0.04))
b2 = tf.Variable(tf.random_normal([10]))
hidden_layer = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
hidden_layer_ = tf.nn.softmax(tf.add(tf.matmul(hidden_layer, W2), b2))
hidden_clipped = tf.clip_by_value(hidden_layer_, 1e-10, 0.9999999)

#calculating cross_entropy
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y_ * tf.log(hidden_clipped)+ (1 - y_) * tf.log(1 - hidden_clipped), axis=1))
#optimizing the weights
optimiser = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(hidden_layer_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#training
epochs=40
with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   total_batch = int(len(mnist.train.labels) / 100)
   for epoch in range(epochs):
       for i in range(total_batch):
           batch_xs, batch_ys = mnist.train.next_batch(100)
           sess.run([optimiser, cross_entropy], feed_dict={x: batch_xs, y_:batch_ys})
   print('mlp with one hidden layer train accuracy:' + str(sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.train.labels})))
   print('mlp with one hidden layer test accuracy(MNIST):' + str(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))
   print('mlp with one hidden layer test accuracy(USPS):' + str(sess.run(accuracy, feed_dict={x: X_images, y_: X_labels})))

#convolutional neural network
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) +
b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})                                                
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5})
    print('CNN test accuracy %g' % accuracy.eval(feed_dict={x: X_images, y_: X_labels, keep_prob: 1.0}))
    print('CNN train accuracy(MNIST) %g' % accuracy.eval(feed_dict={x: mnist.train.images, y_: mnist.train.labels, keep_prob: 1.0}))
    print('CNN test accuracy(MNIST) %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    print('CNN test accuracy(USPS) %g' % accuracy.eval(feed_dict={x: X_images, y_: X_labels, keep_prob: 1.0}))
