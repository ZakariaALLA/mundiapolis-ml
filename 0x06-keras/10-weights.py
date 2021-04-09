#!/usr/bin/env python3
"""
Weights
"""


import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """ 
    The function save_weights
    """
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
    The function load_weights
    """
    network.load_weights(filename)
    return None
