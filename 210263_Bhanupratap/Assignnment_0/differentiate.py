import numpy as np
import matplotlib.pyplot as plt
import activation as act

def sigmoidDerivative(z):
    return (-np.exp(-z)/(1+np.exp(-z))**2)
def tanhDerivative(z):
    return (1+ act.tanh(z))
def ReLUDerivative(z):
    return ((z>0)*1)



