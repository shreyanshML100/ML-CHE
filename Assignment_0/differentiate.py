import matplotlib.pyplot as plt
import numpy as np

# sigmoid function

def sigmoid(x):
    s=1/(1+np.exp(-x))
    ds=s*(1-s)  
    return s,ds

print(sigmoid(x))

# tanh function

def tanh(y):
    t=(np.exp(y)-np.exp(-y))/(np.exp(y)+np.exp(-y))
    dt=1-t**2
    return t,dt

print(tanh(y))

# ReLU function

def ReLU(z):
    R = np.maximum(0,z) 
    dR = 1 if z > 0 else 0
    return R,dR

print(ReLU(z))    
