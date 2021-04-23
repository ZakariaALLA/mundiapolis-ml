#!/usr/bin/env python3
"""
Defines the function that calculates the cost of a neural network
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network
     """
    weights_squared = 0
    for i in range(1, L + 1):
        level_weight = weights["W{}".format(i)]
        # squared = np.matmul(level_weight.transpose(), level_weight)
        weights_squared += np.linalg.norm(level_weight)
    l2_reg_cost = cost + ((lambtha / (2 * m)) * weights_squared)
    return l2_reg_cost
