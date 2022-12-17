import numpy as np
from matplotlib import pyplot as plt
 
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
 
def tanh(z):
    return (np.e**z-np.e**(-1*z))/(np.e**z+np.e**(-1*z))
def ReLU(z):
    return np.maximum(z,0)
def s(z):
    return 1/(1+np.e**(-1*z))
 
z = np.linspace(-100, 100, 100)
plt.plot(z, tanh(z), color='red')
plt.plot(z, s(z), color='red')
plt.plot(z, ReLU(z), color='red')
 
plt.show()