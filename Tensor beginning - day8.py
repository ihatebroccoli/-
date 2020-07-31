# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 18:14:37 2020

@author: grago
"""
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m,n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]
#X = tf.constant(housing_data_plus_bias, dtype = tf.float32, name = "X")
X = tf.placeholder(dtype = tf.float32, name = "X", shape = (None,n + 1))
#y = tf.constant(housing.target.reshape(-1,1), dtype = tf.float32, name = "y")
y = tf.placeholder(dtype = tf.float32, name = "y", shape = (None, 1))
#XT = tf.transpose(X)
#theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)), XT), y)
batch_size = 100
def fetch_batch(epoch, batch_index, batch_size):
    X_batch = housing_data_plus_bias[batch_index * batch_size : (batch_index + 1) * batch_size]
    y_batch = housing.target.reshape(-1,1)[batch_index * batch_size : (batch_index + 1) * batch_size]
    return X_batch, y_batch


theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name= "theta")
y_pred = tf.matmul(X,theta, name = "predictions")
n_epochs = 1000
n_batches = int(np.ceil(m / batch_size)) #1 batch size

learning_rate = 0.01

error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name = "mse")
optimizer = tf.train.AdamOptimizer(learning_rate =learning_rate).minimize(mse)




with tf.Session() as sess:
    #theta_value = theta.eval()
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        #if epoch % 100 == 0:
        #   print("Epochs: ", epoch, " MSE: ", mse.eval())
        #sess.run(optimizer)
        for batch_index in range(n_batches):
            
            
            X_batch, y_batch = fetch_batch(epoch,batch_index, batch_size)
            sess.run(optimizer, feed_dict = {X: X_batch, y: y_batch})
        if epoch % 100 == 0:
            print(theta.eval())