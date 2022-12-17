import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

def ReLU(z):
    return np.maximum(0,z)
        




