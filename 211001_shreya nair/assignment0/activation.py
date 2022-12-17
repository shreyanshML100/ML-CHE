import numpy as np
z=input("Enter value of z:")
def sigmoid(z):
    return 1/(np.exp(-z)+1)

def tanh(z):
   return  (np.exp(z)-np.exp(-z))/(np.exp(-z)+np.exp(z))

def relu(z):
    if z>0:
        return z
    else:
        return 0
