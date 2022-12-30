import numpy as np
z=int(input("z:"))

def sigmoid(z):
    return 1/(1+np.exp(-z))
print(sigmoid(z))
def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
print(tanh(z))
def ReLU(z):
    if (z>0):
        return z
    else:
        return 0
print(ReLU(z))