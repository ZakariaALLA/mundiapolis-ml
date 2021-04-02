#!/usr/bin/env python3

"""  Neuron class  """

import numpy as np


class Neuron:

"""  class Constructor """

    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        else:
            self.W = np.random.normal(size=(1, nx))
            self.b = 0
            self.A = 0
