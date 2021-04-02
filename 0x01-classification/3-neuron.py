#!/usr/bin/env python3
''' Neuron  '''


import numpy as np


class Neuron:
    '''  Neuron class  '''

    def __init__(self, nx):
        '''  Neuron class  '''

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        ''' get W '''
        return (self.__W)

    @property
    def b(self):
        ''' get b '''
        return (self.__b)

    @property
    def A(self):
        ''' get A '''
        return (self.__A)

    def forward_prop(self, X):
        '''The forward function'''
        z = np.matmul(self.W, X) + self.b
        self.__A = 1 / (1 + (np.exp(-z)))
        return (self.A)

    def cost(self, Y, A):
        '''The cost function'''
        m = Y.shape[1]
        m_loss = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = (1 / m) * (-(m_loss))
        return (cost)
