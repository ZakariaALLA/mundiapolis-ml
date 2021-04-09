#!/usr/bin/env python3
"""
Test
"""


import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    The function test_model
    """
    loss, accuracy = network.evaluate(x=data, y=labels, verbose=verbose)
    return loss, accuracy
