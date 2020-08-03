# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 09:50:03 2020

@author: grago
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
X = iris.data[:,(2,3)]
y = (iris.target == 0).astype(np.int)

per_clf = Perceptron(random_state = 42)
per_clf.fit(X,y)

y_pred = per_clf.predict([[2, 0.5]])
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")
X_train = mnist.train.images
y_train = mnist.train.labels
y_train = y_train.astype("int32")
X_test = mnist.test.images
y_test = mnist.test.labels
y_test = y_test.astype("int32")


feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units = [300,100], n_classes = 10, feature_columns = feature_cols)
dnn_clf = tf.contrib.learn.SKCompat(dnn_clf)
dnn_clf.fit(X_train, y_train, batch_size = 50, steps = 40000)

from sklearn.metrics import accuracy_score
y_pred = dnn_clf.predict(X_test)
print(accuracy_score(y_test, y_pred['classes']))
"""

n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")
(X_train,y_train), (X_test,y_test) = tf.keras.datasets.mnist.load_data()
print(X_train.shape, X_test.shape)
X_train = X_train.reshape(60000, 28 * 28)
X_test = X_test.reshape(10000, 28*28)
X_valid , X_train = X_train[:5000], X_train[5000:]
y_valid , y_train = y_train[:5000], y_train[5000:]

X = tf.placeholder(shape = (None,n_inputs), dtype = "float32", name = "X")
y = tf.placeholder(shape = (None), dtype = "int32", name = "y")

def neuron_layer(X,n_neurons, name,activation = None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2/ np.sqrt(n_inputs + n_neurons)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev = stddev)
        W = tf.Variable(init, name = "kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name = "bias")
        Z = tf.matmul(X,W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z
        
with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, name="hidden1", activation = tf.nn.relu)
    hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2", activation = tf.nn.relu)
    logits = neuron_layer(hidden2, n_outputs, name="logits")
    
    
with tf.name_scope("loss"):
    xentrophy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
    loss = tf.reduce_mean(xentrophy, name = "loss")
    
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate =learning_rate)
    training_op = optimizer.minimize(loss)
    
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 40
batch_size = 50

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in  shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict = {X:X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict = {X:X_batch, y: y_batch})
        acc_valid = accuracy.eval(feed_dict = {X:X_valid, y: y_valid})
        print(epoch, "배치 데이터 정확도: ", acc_batch, "검증 데이터 정확도: ", acc_valid)
    save_path = saver.save(sess, "./model_final.ckpt")
