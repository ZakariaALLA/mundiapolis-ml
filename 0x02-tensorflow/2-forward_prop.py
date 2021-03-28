#!/usr/bin/env python3
import tensorflow as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    create_layer = __import__('1-create_layer').create_layer
    for i in range(len(layer_sizes)):
        if i is 0:
            output = create_layer(x, layer_sizes[i], activations[i])
        else:
            output = create_layer(output, layer_sizes[i], activations[i])
    return output
