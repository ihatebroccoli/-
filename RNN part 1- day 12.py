# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 09:01:26 2020

@author: grago
"""

import tensorflow as tf
import numpy as np
n_steps = 2
n_inputs =3
n_neurons = 5
"""
X0 = tf.placeholder(tf.float32, [None,n_inputs])
X1 = tf.placeholder(tf.float32, [None,n_inputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0,X1], dtype = tf.float32)

Y0, Y1 = output_seqs

Wx = tf.Variable(tf.random_normal(shape = [n_inputs, n_neurons], dtype = tf.float32))
Wy = tf.Variable(tf.random_normal(shape = [n_neurons, n_neurons], dtype = tf.float32))
b = tf.Variable(tf.zeros([1, n_neurons], tf.float32))

Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
Y1 = tf.tanh(tf.matmul(X1, Wx) + tf.matmul(Y0, Wy) + b)

seq_length = tf.placeholder(tf.int32, shape =[None])
X = tf.placeholder(tf.float32, [None,n_steps, n_inputs])
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons)
output_seqs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype = tf.float32, sequence_length = seq_length)
output = tf.transpose(tf.stack(output_seqs), perm = [1,0,2])

init = tf.global_variables_initializer()

X_batch = np.array([
        [[0,1,2], [9,8,7]],
        [[3,4,5], [0,0,0]],
        [[6,7,8], [6,5,4]],
        [[9,0,1], [3,2,1]],
    ])
seq_length_batch = np.array([2,1,2,2])


with tf.Session() as sess:
    init.run()
    outputs_val, states_val = sess.run([output,states], feed_dict = {X : X_batch, seq_length : seq_length_batch})
    print(outputs_val)
    

(X_train, y_train),(X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(60000,784)
X_test = X_test.reshape(10000, 784)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]



n_steps = 28
n_inputs = 28
n_outputs = 10
n_neurons = 150

learning_rate = 0.001
X = tf.placeholder(tf.float32, [None,n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype = tf.float32)
logits = tf.layers.dense(states, n_outputs)
xentrophy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = y)

loss = tf.reduce_mean(xentrophy)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()

n_epochs = 100
batch_size = 150

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


with tf.Session() as sess:
    init.run()
    for epochs in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train,batch_size):
            X_batch = X_batch.reshape(-1, n_steps, n_inputs)
            sess.run(train_op, feed_dict ={X:X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict = {X: X_batch, y: y_batch})
        #acc_valid = accuracy.eval(feed_dict = {X: X_valid, y: y_valid})
        print(epochs, " ",acc_batch)
"""

n_neurons = 100
n_steps = 40
n_inputs = 1
n_outputs = 1
X_train = []
y_train = []
for i in range (0, 40):
    X_tmp = np.arange(i * 40, (i + 1) * 40)
    X_train.append(X_tmp)
    
    y_tmp = np.cos(np.arange(i * 40, (i + 1) * 40)*(20*np.pi/1000))
    y_train.append(y_tmp)

y_test = np.cos(np.arange(1601, 2400)*(20*np.pi/1000))[:,None]
X_test = np.arange(1601,2400)
X_train = np.reshape(X_train, (40,40,1))
y_train = np.reshape(y_train, (40,40,1))
print(np.shape(X_train))

X = tf.placeholder(tf.float32, [None,n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None,n_steps, n_outputs])
cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units = n_neurons, activation = tf.nn.relu), output_size = n_outputs)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)
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
    