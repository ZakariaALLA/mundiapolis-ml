#!/usr/bin/env python3
""" 
Train the model

"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None, verbose=True, shuffle=False):
    """
    Train the model
    """
    history = network.fit(x=data, y=labels, batch_size=batch_size, epochs=epochs, alidation_data=validation_data, verbose=verbose, shuffle=shuffle)
    return history
