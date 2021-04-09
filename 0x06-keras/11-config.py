#!/usr/bin/env python3
"""
Config
"""


import tensorflow.keras as K


def save_config(network, filename):
    """
    The function save_config
    """
    json = network.to_json()
    with open(filename, 'w+') as f:
        f.write(json)
    return None


def load_config(filename):
    """
    The function load_config
    """
    with open(filename, 'r') as f:
        json_string = f.read()
    model = K.models.model_from_json(json_string)
    return model
