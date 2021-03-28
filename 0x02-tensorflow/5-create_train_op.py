#!/usr/bin/env python3
import tensorflow as tf


def create_train_op(loss, alpha):
    gradient_descent = tf.train.GradientDescentOptimizer(alpha)
    return (gradient_descent.minimize(loss))
