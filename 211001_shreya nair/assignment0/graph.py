import numpy as np
import matplotlib.pyplot as plt
import math

def sigmoid(a,b):
    x = np.linspace(a, b, 50)
    y = 1/(1+np.exp(-x))
    plt.plot(x,y)
    plt.xlabel('x')
    plt.ylabel('Sigmoid')
    return plt.show()

def tanh(a,b):
    x = np.linspace(a, b, 50)
    y = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    plt.plot(x,y)
    plt.xlabel('x')
    plt.ylabel('Tanh')
    return plt.show()

def ReLU(a,b):
    x = np.linspace(a, b, 50)
    y = np.maximum(x,0)
    plt.plot(x,y)
    plt.xlabel('x')
    plt.ylabel('ReLU')
    return plt.show()
