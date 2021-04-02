#!/usr/bin/env python3
"""
converts a numeric label vector to a one-hot matrix
"""


import numpy as np


def one_hot_decode(one_hot):
    """
    The one_hot_encode function (converts a numeric label vector to a one-hot matrix)
    """
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None
    vector = one_hot.transpose().argmax(axis=1)
    return vector
