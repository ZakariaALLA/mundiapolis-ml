#!/usr/bin/env python3
"""
Defines the function optimize model
"""


import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Defines optimize model
    """
    network.compile(optimizer=K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2), loss='categorical_crossentropy', metrics=['accuracy'])
    return None
