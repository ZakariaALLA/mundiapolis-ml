#!/usr/bin/env python3
"""
Defines the function that calculates the F1 score
"""


import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix
    """
    p = precision(confusion)
    r = sensitivity(confusion)
    F1 = (2 * p * r) / (p + r)
    return F1
