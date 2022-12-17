import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def tanh(z):
    return np.tanh(z) 


def ReLU(z):
    return np.maximum(0,z)   


