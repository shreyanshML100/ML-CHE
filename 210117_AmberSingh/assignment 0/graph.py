
import numpy as np
import matplotlib.pyplot as plt
import math

def sigmoid(a,b):
    x = np.linspace(a, b, 100)
    y = 1/(1+np.exp(-x))
    plt.plot(x,y,color='green')
    plt.xlabel('X')
    plt.ylabel('Sigmoid(X)')
    plt.title('SIGMOID')
    return plt.show()




def tanh(a,b):
    x = np.linspace(a, b, 100)
    z=np.exp(-x)
    s=np.exp(x)
    y=(s-z)/(s+z)
    plt.plot(x,y,color='black')
    plt.xlabel('X')
    plt.ylabel('tanh(X)')
    plt.title('TANH')
    return plt.show()





def ReLU(a,b):
    x = np.linspace(a, b, 100)
    y = np.maximum(x,0)
    plt.plot(x,y)
    plt.xlabel('X')
    plt.ylabel('ReLU(X)')
    plt.title('ReLU')
    return plt.show()

