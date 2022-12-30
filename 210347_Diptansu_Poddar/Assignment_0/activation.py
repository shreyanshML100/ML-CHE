import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return 1/(np.exp(-z)+1)

def tanh(z):
   return  (np.exp(z)-np.exp(-z))/(np.exp(-z)+np.exp(z))

def relu(z):
    if z>0:
        return z
    else:
        return 0

ReLU = np.vectorize(relu,otypes=[np.float64])
#print((ReLU(1.23)))
