#!/usr/bin/env python3
"""
Calculate loss
"""


import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Function to calculates the softmax
    """
    loss = tf.losses.softmax_cross_entropy(y, logits=y_pred)
    return loss
