#!/usr/bin/env python3
"""
Predict
"""


import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    The function predict
    """
    prediction = network.predict(x=data, verbose=verbose)
    return prediction
