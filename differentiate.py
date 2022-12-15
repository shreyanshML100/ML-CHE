import numpy as np
import matplotlib.pyplot as plt
import activations as act

def sig_deriv(z):
    return -np.exp(-z)/(1+np.exp(-z))^2
def tanh_deriv(z):
    return 1+ act.tanh(z)
def ReLU_deriv(z):
    return (z>0)*1
