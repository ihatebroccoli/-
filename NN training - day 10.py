# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:41:45 2020

@author: grago
"""
import tensorflow as tf
import numpy as np

X = tf.placeholder(dtype = tf.float32, shape = (None,784), name = 'X')
y = tf.placeholder(dtype = tf.int32, shape = (None), name = 'y')
training = tf.placeholder_with_default(False, shape = (), name = 'training')

(X_train, y_train),(X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(60000,784)
X_test = X_test.reshape(10000, 784)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
#he_init = tf.variance_scaling_initializer()
#hidden1 = tf.layers.dense(X, n_hidden1, activation = tf.nn.elu, kernel_initializer=he_init, name = "hidden1")

from functools import partial

my_batch_norm_layer = partial(tf.layers.batch_normalization, training = training, momentum = 0.9)

#my_dense_layer = partial(tf.layers.dense, activation = tf.nn.relu, kernel_regularizer = tf.contrib.layers.l1_regularizer(0.001))
#reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#loss = tf.add_n([base_loss] + reg_losses, name = "loss")

dropout_rate = 0.5
X_drop = tf.layers.dropout(X, dropout_rate, training = training)
"""
def max_norm_regulation(threshold, axes= 1, name= "max_norm", collection = "max_norm"):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm = threshold, axes = axes)
        clip_weights = tf.assign(weights, clipped, name= name)
        tf.add_to_collection(collection, clip_weights)
        return None
    return max_norm
"""
#threshold = 1.0
#weights = tf.get_default_graph().get_tensor_by_name("hidden/kernel:0")
#clipped_weight = tf.clip_by_norm(weights, clip_norm = threshold, axes = 1)


hidden1 = tf.layers.dense(X, n_hidden1, name = "hidden1")
# hidden_drop = tf.layers.dropout(hidden1, dropout_rate, training = training)
bn1 = my_batch_norm_layer(hidden1)
bn1_act = tf.nn.elu(bn1)

hidden2 = tf.layers.dense(bn1_act, n_hidden2, name = "hidden2")
bn2 = my_batch_norm_layer(hidden2)
bn2_act = tf.nn.elu(bn2)

logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name = "logits")
logits =my_batch_norm_layer(logits_before_bn)

xentrophy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = y)
loss = tf.reduce_mean(xentrophy, name = "loss")

correct = tf.nn.in_top_k(logits,y, 1)
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


saver = tf.train.Saver()
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
training_op = optimizer.minimize(loss)

n_epochs = 40
batch_size = 50

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train,y_train,batch_size):
           sess.run([training_op, extra_update_ops], feed_dict = {training: True, X : X_batch, y : y_batch})
        accuracy_val = accuracy.eval(feed_dict = {X: X_valid, y : y_valid})
        print(epoch, "검증 데이터 정확도: ", accuracy_val)
    save_path = saver.save(sess, "./model_final.ckpt")

