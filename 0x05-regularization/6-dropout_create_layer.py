#!/usr/bin/env python3
"""
DropOut Regularization
"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Creates a tensorflow layer that includes dropout regularization.
    """
    dropout = tf.layers.Dropout(keep_prob)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=dropout)
    return layer(prev)
