# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 14:24:42 2020

@author: grago
"""
import numpy as np
import tensorflow as tf

nan= np.nan
T = np.array([
              [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
              [[0.0, 1.0, 0.0], [nan, nan, nan], [0.0, 0.0, 1.0]],
              [[nan, nan, nan], [0.8, 0.1, 0.1], [nan, nan, nan]],
        ])

R = np.array([
               [[10., 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
               [[0.0, 0.0, 0.0], [nan, nan, nan], [0.0, 0.0, -50.]],
               [[nan, nan, nan], [40., 0.0, 0.0], [nan, nan, nan]],
        ])

possible_actions = [[0, 1, 2], [0, 2], [1]]
learning_rate0 = 0.05
learning_rate_decay = 0.1
n_iterations = 20000
discount_factor = 0.90

s = 0

Q = np.full((3,3), -np.inf)
for state, actions in enumerate(possible_actions):
    Q[state, actions]  = 0.0
for iteration in range(n_iterations):
    a = np.random.choice(possible_actions[s])
    sp = np.random.choice(range(3), p = T[s,a])
    reward = R[s, a, sp]
    learning_rate = learning_rate0 / (1 + iteration * learning_rate_decay)
    Q[s, a] = ((1 - learning_rate) * Q[s, a] + learning_rate * (reward + discount_factor * np.max(Q[sp])))
print (Q)
    
"""
    discount_factor = 0.95
    n_iterations = 100
    
    for iteration in range(n_iterations):
        Q_prev = Q.copy()
        for s in range(3):
            for a in possible_actions[s]:
                Q[s, a] = np.sum([
                    T[s, a, sp] * (R[s, a, sp] + discount_factor * np.max(Q_prev[sp]))
                    for sp in range(3)    
                    ])
print (Q)
print (np.argmax(Q, axis = 1))
"""