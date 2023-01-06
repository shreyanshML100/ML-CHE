import numpy as np
import matplotlib as plt

def sigmoid(z):
    return 1/(1+np.exp(-z))
def tanh(Z):
    return (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+(np.exp(-Z)))
def Relu(Z):
    return (np.maximum(0,Z))
