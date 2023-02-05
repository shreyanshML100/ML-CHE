import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import activation

h=0.00001 #step size

def central(f, x):
    '''
    O(h^4)-accurate central difference approximation
    '''
    return ( (f(x-2*h)) - 8 * (f(x-h)) + 8 * (f(x+h)) - (f(x+2*h)) ) / (12*h)

def forward(f, x):
    '''
    O(h^2)-accurate forward difference approximation
    '''
    return (-3 * f(x) + 4 * f(x+h) - f(x+2*h)) / (2*h)
    
def backward(f, x):
    '''
    O(h^2)-accurate backward difference approximation
    '''
    return (3 * f(x) - 4 * f(x-h) + f(x-2*h)) / (2*h)
    

def sigmoid(z):
    return central(activation.sigmoid, z)

def tanh(z):
    return central(activation.tanh, z)

def ReLU(z):
    '''
    Handles non-differentiability at 0 by 
    returning forward / backward approximations
    '''
    return np.where(z<=0, backward(activation.ReLU, z), forward(activation.ReLU, z))
    #if z<=0 returns backward difference, else returns forward difference
