import numpy as np
import matplotlib.pyplot as plt

from activation import *
from scipy.interpolate import interp1d

def sigmoidG(a, b):
    x = np.arange(a, b, 0.01)
    y = np.empty(len(x))
    for i in range(0, len(x)):
        y[i] = sigmoid(x[i])

    cubic_interpolation_model = interp1d(x, y, kind = "cubic")
 
    X_=np.linspace(x.min(), x.max(), 500)
    Y_=cubic_interpolation_model(X_)
     
    plt.plot(X_, Y_)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def tanhG(a, b):
    x = np.arange(a, b, 0.01)
    y = np.empty(len(x))
    for i in range(0, len(x)):
        y[i] = tanh(x[i])

    cubic_interpolation_model = interp1d(x, y, kind = "cubic")
 
    X_=np.linspace(x.min(), x.max(), 500)
    Y_=cubic_interpolation_model(X_)
     
    plt.plot(X_, Y_)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    
def ReLUG(a, b):
    
    x = np.arange(a, b, 0.01)
    y = np.empty(len(x))
    for i in range(0, len(x)):
        y[i] = ReLU(x[i])

    cubic_interpolation_model = interp1d(x, y, kind = "cubic")
 
    X_=np.linspace(x.min(), x.max(), 500)
    Y_=cubic_interpolation_model(X_)
     
    plt.plot(X_, Y_)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

