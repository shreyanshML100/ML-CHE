import numpy as np
import matplotlib as plt

## Defining functions

def sigmoid(z):
    return 1/(1+np.exp(-z))
def tanh(Z):
    return (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+(np.exp(-Z)))
def Relu(Z):
    return (np.maximum(0,Z))

##Main 

    #print(sigmoid(5)) 
    #print(tanh(5))
    #print(Relu(6))