# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import numpy as np
import os
from PIL import Image
import glob
import tensorflow as tf


X_usps = []
Y_usps = []

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
            X_usps.append(pic)
            ans[i] = 1
            Y_usps.append(ans)
          
            
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.04))
b1 = tf.Variable(tf.random_normal([300]))
W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.04))
b2 = tf.Variable(tf.random_normal([10]))
hidden_layer = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
hidden_layer_ = tf.nn.softmax(tf.add(tf.matmul(hidden_layer, W2), b2))
print(hidden_layer_.shape)
hidden_clipped = tf.clip_by_value(hidden_layer_, 1e-10, 0.9999999)

#calculating cross_entropy
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y_ * tf.log(hidden_clipped)+ (1 - y_) * tf.log(1 - hidden_clipped), axis=1))

def backprop(W1,W2):
    hidden_layer = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
    hidden_layer_ = tf.nn.softmax(tf.add(tf.matmul(hidden_layer, W2), b2))
    hidden_clipped = tf.clip_by_value(hidden_layer_, 1e-10, 0.9999999)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y_ * tf.log(hidden_clipped)+ (1 - y_) * tf.log(1 - hidden_clipped), axis=1))
    error=y_-hidden_layer_
    delta=error*tf.transpose((hidden_layer_*(1-hidden_layer_)))
    hidden_error=tf.matmul(delta,tf.transpose(W2))
    hidden_delta=tf.matmul((hidden_layer_*(1-hidden_layer_)),hidden_error)
    W1+=tf.matmul(tf.transpose(x),hidden_delta)
    delta_new=tf.matmul(tf.matmul(tf.transpose(hidden_layer),error),tf.transpose(hidden_layer_))
    W2+=tf.matmul(delta_new,hidden_layer_)
    return W1,W2,cross_entropy
    

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(hidden_layer_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
optimiser = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


epochs=10
with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   total_batch = int(len(mnist.train.labels) / 100)
   for epoch in range(epochs):
       for i in range(total_batch):
		   W1,W2,cross_entropy=backprop(W1,W2)
           hidden_layer = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
           hidden_layer_ = tf.nn.softmax(tf.add(tf.matmul(hidden_layer, W2), b2))
           correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(hidden_layer_, 1))
           accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
           batch_xs, batch_ys = mnist.train.next_batch(100)
           sess.run([optimiser, cross_entropy], feed_dict={x: batch_xs, y_:batch_ys})
   print(sess.run(accuracy, feed_dict={x: X_usps, y_: Y_usps}))
