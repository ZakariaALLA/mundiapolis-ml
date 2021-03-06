#!/usr/bin/env python3
''' Neuron  '''


import numpy as np


class Neuron:
    '''  Neuron class  '''

    def __init__(self, nx):
        ''' class Constructor '''

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        ''' get W '''
        return self.__W

    @property
    def b(self):
        ''' get b '''
        return self.__b

    @property
    def A(self):
        ''' get A '''
        return self.__A

    def forward_prop(self, X):
        '''The forward function'''
        x = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-x))
        return self.__A
