#!/usr/bin/env python3
"""
Defines the function that calculates the precision
"""


import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix
    """
    classes = confusion.shape[0]
    precision = []
    for column in range(classes):
        correct = 0
        total = 0
        for row in range(classes):
            if row == column:
                correct += confusion[row][column]
            total += confusion[row][column]
        precision.append(correct / total)
    return np.asarray(precision)
