#!/usr/bin/env python3
import tensorflow as tf


def create_layer(prev, n, activation):
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation=activation, name="layer", kernel_initializer=weights_initializer)
    return (layer(prev))
