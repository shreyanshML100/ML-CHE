import numpy as np
import matplotlib.pyplot as plt

def sigmoid(a,b):
    x = np.linspace(a, b, 10)
    y = 1/(1+np.exp(-x))
    plt.plot(x,y)
    plt.xlabel('X')
    plt.ylabel('Sigmoid(X)')
    return plt.show()

def tanh(a,b):
    x = np.linspace(a, b, 10)
    y = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    plt.plot(x,y)
    plt.xlabel('X')
    plt.ylabel('Tanh(X)')
    return plt.show()

def ReLU(a,b):
    x = np.linspace(a, b, 10)
    y = np.maximum(x,0)
    plt.plot(x,y)
    plt.xlabel('X')
    plt.ylabel('ReLU(X)')
    return plt.show()
