# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 16:13:50 2020

@author: grago
"""
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt
china = load_sample_image("china.jpg")
flower = load_sample_image("flower.jpg")
dataset = np.array([china,flower], dtype = "float32")
batch_size, height, width, channels = dataset.shape

filters = np.zeros(shape = (7,7,channels,2), dtype = np.float)
filters[:,3,:,0] = 1
filters[3,:,:,1] = 1

X = tf.placeholder(tf.float32, shape = (None, height, width, channels))

convolution = tf.nn.conv2d(X,filters,strides = [1,2,2,1], padding = "SAME")
conv = tf.layers.conv2d(X, filters = 2, kernel_size=7, strides = [2,2], padding = "SAME")
max_pool = tf.nn.max_pool(X, [1,2,2,1], [1,2,2,1], padding = "VALID")
with tf.Session() as sess:
    #output = sess.run(convolution, feed_dict = {X: dataset})
    output = sess.run(max_pool, feed_dict = {X: dataset})

#plt.imshow(output[0,:,:,1], cmap = "gray")
plt.imshow(output[0].astype(np.uint8))
plt.show()


