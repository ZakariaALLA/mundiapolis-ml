#!/usr/bin/env python3
"""
Defines one hot function
"""


import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    One hot
    """
    one_hot = K.utils.to_categorical(labels, num_classes=classes)
    return one_hot
