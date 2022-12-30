import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import activation

def sigmoid(a,b):
    x=np.linspace(a,b,1000)
    y=activation.sigmoid(x)
    plt.plot(x,y)
    plt.title("Sigmoid Function")
    plt.show()

def tanh(a,b):
    x=np.linspace(a,b,1000)
    y=activation.tanh(x)
    plt.plot(x,y)
    plt.title("tanh Function")
    plt.show()

def ReLU(a,b):
    x=np.linspace(a,b,1000)
    y=activation.ReLU(x)
    plt.plot(x,y)
    plt.title("ReLU Function")
    plt.show()

    

