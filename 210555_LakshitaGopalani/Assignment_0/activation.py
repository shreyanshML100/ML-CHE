import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1+np.exp(-z))

def tanh(z):
    return((np.exp(2*z)-1)/(np.exp(2*z)+1))

def ReLU(z):
    return np.maximum(0.0,z)
