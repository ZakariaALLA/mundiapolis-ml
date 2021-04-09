#!/usr/bin/env python3
"""
Defines the save_model function
"""


import tensorflow.keras as K


def save_model(network, filename):
    """
    The function save_model
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    The function load_model
    """
    model = K.models.load_model(filename)
    return model
