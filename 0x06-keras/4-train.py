#!/usr/bin/env python3
"""
Defines train model function
"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, verbose=True, shuffle=False):
    """
    Train model
    """
    history = network.fit(x=data, y=labels, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle)
    return history
