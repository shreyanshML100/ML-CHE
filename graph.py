import numpy as np
import matplotlib.pyplot as plt
z = np.arange(-100, 50, 0.001)
def sigmoid_function():
    y=1/(1+np.exp(-z))
    plt.plot(z,y)
    plt.grid()
    plt.show()
sigmoid_function()
def tanh():
    y=(np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
    plt.plot(z,y)
    plt.grid()
    plt.show()
tanh()
def ReLU():
    if z>0:
        y=z
    else:
        y=0
    plt.plot(z, y)
    plt.grid()
    plt.show()
ReLU()

