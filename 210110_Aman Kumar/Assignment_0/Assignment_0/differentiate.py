import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def diff_sig(sigmoid,z0,h):
    return (sigmoid(z0+h) - sigmoid(z0-h))/(2*h)
print(diff_sig(sigmoid,0,1))

def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
def diff_tan(tanh,z0,h):
    return (tanh(z0+h) - tanh(z0-h))/(2*h)
print(diff_tan(tanh,0,1))

def ReLU(z):
    if (z>0):
        return z
    else:
        return 0
def diff_rel(ReLU,z0,h):
    return (ReLU(z0+h) - ReLU(z0-h))/(2*h)
print(diff_rel(ReLU,0,1))
