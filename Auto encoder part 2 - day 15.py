# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 10:12:07 2020

@author: grago
"""
import tensorflow as tf
import os

from functools import partial
import numpy as np
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

"""
activation = tf.nn.elu
regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
initializer = tf.variance_scaling_initializer()

noise_level = 1.0
dropout_rate = 0.3


X = tf.placeholder(dtype = tf.float32, shape = (None,n_inputs), name = 'X')
X_noisy = X + noise_level * tf.random_normal(tf.shape(X))

weights1_init = initializer([n_inputs, n_hidden1])
weights2_init = initializer([n_hidden1, n_hidden2])
weights1 = tf.Variable(weights1_init, dtype = tf.float32, name = "weights1")
weights2 = tf.Variable(weights2_init, dtype = tf.float32, name = "weights2")
weights3 = tf.Variable(tf.transpose(weights2, name= "weights3"))
weights4 = tf.Variable(tf.transpose(weights1, name= "weights4"))

biases1 = tf.Variable(tf.zeros(n_hidden1), name = "biases1")
biases2 = tf.Variable(tf.zeros(n_hidden2), name = "biases2")
biases3 = tf.Variable(tf.zeros(n_hidden3), name = "biases3")
biases4 = tf.Variable(tf.zeros(n_outputs), name = "biases4")

hidden1 = activation(tf.matmul(X_noisy,weights1) + biases1)
hidden2 = activation(tf.matmul(hidden1,weights2) + biases2)
hidden3 = activation(tf.matmul(hidden2,weights3) + biases3)
outputs = (tf.matmul(hidden3,weights4) + biases4)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
reg_losses = regularizer(weights1) + regularizer(weights2)
loss = reconstruction_loss + reg_losses

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

with tf.name_scope("phase1"):
    phase1_outputs = tf.matmul(hidden1, weights4) + biases4
    phase1_reconstruction_loss = tf.reduce_mean(tf.square(phase1_outputs - X))
    phase1_reg_loss = regularizer(weights1) + regularizer(weights4)
    phase1_loss = phase1_reconstruction_loss + phase1_reg_loss
    phase1_training_op = optimizer.minimize(phase1_loss)
    
    phase1_correct = tf.equal(tf.argmax(outputs, 1), tf.argmax(X, 1))
    phase1_accuracy = tf.reduce_mean(tf.cast(phase1_correct ,dtype = tf.float32))
with tf.name_scope("phase2"):
    phase2_reconstruction_loss = tf.reduce_mean(tf.square(hidden3 - hidden1))
    phase2_reg_loss = regularizer(weights2) + regularizer(weights3)
    phase2_loss = phase2_reconstruction_loss + phase2_reg_loss
    train_vars = [weights2, biases2, weights3, biases3]
    phase2_training_op = optimizer.minimize(phase2_loss, var_list = train_vars)


init = tf.global_variables_initializer()


n_epochs_1 = 5
n_epochs_2 = 10
batch_size = 50

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx],y[batch_idx]
        yield X_batch, y_batch

with tf.Session() as sess:
    init.run()
    for epoch in range (n_epochs_1):
        n_batches = len(X_train)
        for iterations in range(n_batches):
            X_batch, y_batch = next (shuffle_batch(X_train, y_train, batch_size))
            sess.run(phase1_training_op, feed_dict = {X: X_batch})
            if iterations % 100 == 0:
                print(phase1_accuracy.eval( feed_dict = {X: X_batch}))
    for epoch in range (n_epochs_2):
        n_batches = len(X_train)
        for iterations in range(n_batches):
            X_batch, y_batch = next (shuffle_batch(X_train, y_train, batch_size))
            sess.run(phase2_training_op, feed_dict = {X: X_batch})


def kl_divergence(p,q):
    return p * tf.log(p/q) + (1-p) * tf.log((1-p)/(1-q))
learning_rate = 0.01

sparsity_target = 0,1
sparsity_weight = 0.2

he_init = tf.variance_scaling_initializer()
hidden1 = tf.layers.dense(X, n_hidden1, activation= tf.nn.sigmoid, kernel_initializer = he_init)
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation= tf.nn.elu, kernel_initializer = he_init)
hidden3 = tf.layers.dense(hidden2, n_hidden3, activation= tf.nn.elu, kernel_initializer = he_init)
logits = tf.layers.dense(hidden3, n_outputs)
outputs = tf.nn.sigmoid(logits)

hidden1_mean = tf.reduce_mean(hidden1, axis = 0)
sparsity_loss = tf.reduce_sum(kl_divergence(sparsity_target, hidden1_mean))
xentrophy = tf.nn.sigmoid_cross_entropy_with_logits(labels = X, logits = logits)
reconstruction_loss = tf.reduce_sum(xentrophy)
loss = sparsity_weight * sparsity_loss + reconstruction_loss
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(loss)

"""
n_inputs = 28*28
n_hidden1 = 500
n_hidden2 = 500
n_hidden3 = 20
n_hidden4 = n_hidden2
n_hidden5 = n_hidden1
n_outputs = n_inputs
learning_rate = 0.001

initializer = tf.variance_scaling_initializer()
my_dense_layer = partial(tf.layers.dense,
                         activation = tf.nn.elu,
                         kernel_initializer = initializer)

X = tf.placeholder(dtype = tf.float32, shape = [None,n_inputs])
hidden1 = my_dense_layer(X, n_hidden1)
hidden2 = my_dense_layer(hidden1, n_hidden2)
hidden3_mean = my_dense_layer(hidden2, n_hidden3, activation = None)
hidden3_gamma = my_dense_layer(hidden2, n_hidden3, activation = None)
noise = tf.random_normal(tf.shape(hidden3_gamma), dtype = tf.float32)
hidden3 = hidden3_mean + tf.exp(0.5 * hidden3_gamma) * noise
hidden4 = my_dense_layer(hidden3, n_hidden4)
hidden5 = my_dense_layer(hidden4, n_hidden5)

logits = my_dense_layer(hidden5, n_outputs, activation = None)
outputs = tf.sigmoid(logits)

xentrophy= tf.nn.sigmoid_cross_entropy_with_logits(logits= logits, labels = X)
reconstruction_loss = tf.reduce_mean(xentrophy)
latent_loss = 0.5 * tf.reduce_sum(tf.exp(hidden3_gamma) + tf.square(hidden3_mean) - 1 - hidden3_gamma)
loss = reconstruction_loss + latent_loss

optimizer = tf.train.AdamOptimizer(learning_rate =learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_digits = 60
n_epochs = 2
batch_size = 150

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

def plot_image(image, shape = [28,28]):
    plt.imshow(image.reshape(shape), cmap = "Greys", interpolation= "nearest")
    plt.axis("off")
    
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = len(X_train)
        print(epoch)
        for iteration in range(n_batches):
            X_batch, y_batch = next(shuffle_batch(X_train, y_train, batch_size))
            sess.run(training_op, feed_dict = {X: X_batch})
    codings_rnd = np.random_normal(size = [n_digits, n_hidden3])
    outputs_val = outputs.eval(feed_dict = {hidden3 : codings_rnd})
    for iteration in range(n_digits):
        plt.subplot(n_digits, 10, iteration + 1)
        plot_image(outputs_val[iteration])