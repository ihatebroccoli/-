import numpy as np
import gym
# -*- coding: utf-8 -*-

env = gym.make("CartPole-v0")
obs = env.reset()
"""
img = env.render(mode = "rgb_array")

action = 1
obs, reward, done, info = env.step(action)
print(obs, reward, done, info)

def basic_policy(obs):
    angle= obs[2]
    return 0 if angle <0 else 1
totals = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for steop in range(1000):
        action = basic_policy(obs)
        obs,reward,done,info = env.step(action)
        episode_rewards += reward
        if done:
            break
    totals.append(episode_rewards)

print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))
"""

import tensorflow as tf

n_inputs = 4
n_hidden = 4
n_outputs = 1
initializer = tf.variance_scaling_initializer()

X = tf.placeholder(tf.float32, [None, n_inputs])
hidden = tf.layers.dense(X, n_hidden, activation = tf.nn.elu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
outputs = tf.nn.sigmoid(logits)
p_left_and_right = tf.concat(axis = 1, values = [outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples = 1)

y = 1. - tf.to_float(action)
learning_rate = 0.01
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = y)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad, variable in grads_and_vars]
gradient_placeholders = []
grads_and_vars_feed =[]

for grad, variable in grads_and_vars:
    if np.shape(grad) == None:
        continue
    graident_placeholder = tf.placeholder(tf.float32, shape = np.shape(grad))
    gradient_placeholders.append(graident_placeholder)
    grads_and_vars_feed.append((graident_placeholder, variable))
    
training_op = optimizer.apply_gradients(grads_and_vars_feed)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

def discount_rewards(rewards, discount_factor):
    discount_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_factor
        discount_rewards[step] = cumulative_rewards
    return discount_rewards

def discount_and_normalize_rewards(all_rewards, discount_factor):
    all_discounted_rewards = [discount_rewards(rewards, discount_factor)
                              for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/ reward_std
            for discounted_rewards in all_discounted_rewards]

n_iteration = 250
n_max_steps = 1000
n_games_per_update = 10
save_iterations = 10
discount_factor = 0.95

with tf.Session() as sess:
    init.run()
    for iteration in range (n_iteration):
        all_rewards = []
        all_gradients = []
        for game in range(n_games_per_update):
            current_rewards = []
            current_gradients = []
            obs = env.reset()
            for step in range(n_max_steps):
                action_val, gradient_val = sess.run(
                        [action, gradients],
                        feed_dict = {X: obs.reshape(1, n_inputs)})
                obs,reward,done,info = env.step(action_val[0][0])
                current_rewards.append(reward)
                current_gradients.append(gradient_val)
                if done:
                    break
                all_rewards.append(current_rewards)
                all_gradients.append(current_gradients)
        all_rewards = discount_and_normalize_rewards(all_rewards, discount_factor)
        feed_dict = {}
        for var_index, grad_placeholder in enumerate(gradient_placeholders):
            mean_gradients = np.mean(
                [reward*all_gradients[game_index][step][var_index]
                    for game_index, rewards in enumerate(all_rewards)
                    for step, reward in enumerate(rewards)],
                axis = 0)
            feed_dict[grad_placeholder] = mean_gradients
        sess.run(training_op, feed_dcit = feed_dict)