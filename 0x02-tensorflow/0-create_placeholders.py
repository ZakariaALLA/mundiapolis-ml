#!/usr/bin/env python3
"""
Create placeholders
"""


import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Function to returns two placeholders
    """
    x = tf.placeholder("float", shape=(None, nx), name="x")
    y = tf.placeholder("float", shape=(None, classes), name="y")
    return x, y
