#!/usr/bin/env python3
"""
Defines function that conducts forward propagation using Dropout
"""


import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout
    """
    outputs = {}
    outputs["A0"] = X
    for index in range(L):
        weight = weights["W{}".format(index + 1)]
        bias = weights["b{}".format(index + 1)]
        z = np.matmul(weight, outputs["A{}".format(index)]) + bias
        dropout = np.random.binomial(1, keep_prob, size=z.shape)
        if index != (L - 1):
            A = np.tanh(z)
            A *= dropout
            A /= keep_prob
            outputs["D{}".format(index + 1)] = dropout
        else:
            A = np.exp(z)
            A /= np.sum(A, axis=0, keepdims=True)
        outputs["A{}".format(index + 1)] = A
    return outputs
