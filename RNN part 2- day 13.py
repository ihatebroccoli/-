# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 19:07:17 2020

@author: grago
"""
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt



class DeviceCellWrapper(tf.contrib.rnn.RNNCell):
    def __init__(self, device, cell):
        self._cell = cell
        self.device = device
    
    @property
    def state_size(self):
        return self._cell.state_size
    
    @property
    def output_size(self):
        return self._cell.output_size
    
    def __call__(self, inputs, state, scope = None):
        with tf.device(self._device):
            return self._cell(inputs, state, scope)


n_neurons = 100
n_steps = 40
n_inputs = 1
n_outputs = 1
n_layers = 20

X_train = []
y_train = []
X_test = []
y_test = []
for i in range (0, 40):
    j = i + 40
    X_tmp = np.arange(i * 40, (i + 1) * 40)
    X_train.append(X_tmp)
    
    y_tmp = np.cos(np.arange(i * 40, (i + 1) * 40)*(20*np.pi/1000))
    y_train.append(y_tmp)
    
    X_tmp = np.arange(j * 40, (j + 1) * 40)
    X_test.append(X_tmp)
    
    y_tmp = np.cos(np.arange(j * 40, (j + 1) * 40)*(20*np.pi/1000))
    y_test.append(y_tmp)


X_train = np.reshape(X_train, (40,40,1))
y_train = np.reshape(y_train, (40,40,1))

X_test = np.reshape(X_test, (40,40,1))
y_test = np.reshape(y_test, (40,40,1))


X = tf.placeholder(tf.float32, [None,n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None,n_steps, n_outputs])

cell = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons, activation = tf.nn.elu)
cells_drop = tf.contrib.rnn.DropoutWrapper(cell,input_keep_prob = 0.5)
layers = [tf.contrib.rnn.OutputProjectionWrapper(cells_drop, output_size = n_outputs) for layer in range(n_layers)]

multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers, state_is_tuple = False)
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype = tf.float32)
learning_rate = 0.001

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op =optimizer.minimize(loss)
init = tf.global_variables_initializer()

n_iteration =1500


with tf.Session() as sess:
    init.run()
    for iteration in range (n_iteration):
        sess.run(train_op, feed_dict = {X : X_train, y: y_train})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict = {X: X_train, y: y_train})
            print(iteration, "\t", mse)
    y_pred = sess.run(outputs, feed_dict = {X : X_test, y: y_test})
    y_grap = np.cos(np.arange(1600, 3200)*(20*np.pi/1000))[:,None]
    X_grap = np.arange(1600,3200)
    print(y_pred, y_pred.shape)
    y_pred = np.reshape(y_pred, (1600))
    plt.plot(X_grap, y_pred, color = 'green')
    plt.plot(X_grap, y_grap, color = 'yellow')
    plt.show()