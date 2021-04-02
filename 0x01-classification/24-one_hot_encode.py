#!/usr/bin/env python3
"""
converts a numeric label vector to a one-hot matrix
"""


import numpy as np


def one_hot_encode(Y, classes):
    """
    The one_hot_encode function (converts a numeric label vector to a one-hot matrix)
    """
    if type(Y) is not np.ndarray:
        return None
    if type(classes) is not int:
        return None
    try:
        one_hot = np.eye(classes)[Y].transpose()
        return one_hot
    except Exception as err:
        return None
