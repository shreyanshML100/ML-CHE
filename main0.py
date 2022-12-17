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
 
    
x=np.array([-100, -15, -9, -5, -1, 0, 1, 5 , 9, 15, 100])
print(s(x))
print(tanh(x))
print(ReLU(x))
 
z = np.linspace(-10, 10, 100)
plt.plot(z, tanh(z), color='red')
plt.plot(z, s(z), color='red')
plt.plot(z, ReLU(z), color='red')
 
plt.show()
