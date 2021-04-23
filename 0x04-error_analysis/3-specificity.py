#!/usr/bin/env python3
"""
Defines the function that calculates the specificity
"""


import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix
    """
    classes = confusion.shape[0]
    specificity = []
    for actual_class in range(classes):
        true_negative = 0
        total = 0
        for row in range(classes):
            if row == actual_class:
                continue
            for column in range(classes):
                if column != actual_class:
                    true_negative += confusion[row][column]
                total += confusion[row][column]
        specificity.append(true_negative / total)
    return np.asarray(specificity)
