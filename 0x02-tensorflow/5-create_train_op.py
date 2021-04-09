#!/usr/bin/env python3
"""
Create train op
"""


import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Function to Create the training operation
    """
    gradient_descent = tf.train.GradientDescentOptimizer(alpha)
    return (gradient_descent.minimize(loss))
