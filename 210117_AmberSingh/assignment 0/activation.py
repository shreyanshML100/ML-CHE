
import numpy as np
def sigm(t):
    z=np.exp(-t)
    sig=1/(1+z)
    return sig

def tanh(t):
    z=np.exp(-t)
    y=np.exp(t)
    tan=(y-z)/(y+z)
    return tan

def ReLU(t):
    return np.maximum(t,0)
