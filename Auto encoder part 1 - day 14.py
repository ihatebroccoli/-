# -*- coding: utf-8 -*-
import tensorflow as tf
"""
n_inputs = 3
n_hidden = 2
n_outputs = n_inputs

learning_rate = 0.01

X = tf.placeholder(dtype = tf.float32, shape = [None, n_inputs])
hidden = tf.layers.dense(X, n_hidden)
outputs = tf.layers.dense(hidden, n_outputs)

reconstruction_loss = tf.reduce_max(tf.square(outputs - X))

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(reconstruction_loss)



init = tf.global_variables_initializer()

from sklearn.datasets import load_iris
iris = load_iris()
X_train = iris.data[:130, :3]

X_test = iris.data[130:, :3]


n_iterations = 1000
codings = hidden

with tf.Session() as sess:
    init.run()
    for iterations in range (n_iterations):
        train_op.run(feed_dict = {X: X_train})
        codings_val = codings.eval(feed_dict = {X: X_test})
        
"""
from functools import partial
import numpy as np
n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 150
n_hidden3 = 300
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.0001

X = tf.placeholder(dtype = tf.float32, shape = (None,784), name = 'X')

training = tf.placeholder_with_default(False, shape = (), name = 'training')

(X_train, y_train),(X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(60000,784)
X_test = X_test.reshape(10000, 784)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

he_init = tf.variance_scaling_initializer()
l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
my_dense_layer = partial(tf.layers.dense, activation = tf.nn.relu, kernel_initializer = he_init, kernel_regularizer =l2_regularizer)

hidden1= my_dense_layer(X, n_hidden1)
hidden2= my_dense_layer(hidden1, n_hidden2)
hidden3= my_dense_layer(hidden2, n_hidden3)
outputs= my_dense_layer(hidden3, n_outputs, activation = None)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses)

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

n_epochs = 5
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range (n_epochs):
        n_batches = len(X_train)
        for iterations in range(n_batches):
            X_batch, y_batch = next (shuffle_batch(X_train, y_train, batch_size))
            sess.run(train_op, feed_dict = {X: X_batch})
        
        