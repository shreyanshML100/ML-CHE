import numpy as np
from numpy import array, exp
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def tanh(y):
    return np.tanh(y)  

def ReLU(x):
    return np.maximum(0,x)    

z = np.linspace(-10, 10)

plot1 = plt.plot(z, sigmoid(z))
plot1 = plt.xlabel('z')
plot1 = plt.ylabel('sigmoid(z)')
plot1 = plt.title('Sigmoid Function in Matplotlib')

plot1 = plt.show()
  

y = np.linspace(-10, 10)

plot2 = plt.plot(y, tanh(y))
plot2 = plt.xlabel('y')
plot2 = plt.ylabel('tanh(y)')
plot2 = plt.title('tanh Function in Matplotlib')

plot2 = plt.show()


x = np.linspace(-10, 10)

plot3 = plt.plot(x, ReLU(x))
plot3 = plt.xlabel('x')
plot3 = plt.ylabel('ReLU(x)')
plot3 = plt.title('ReLU Function in Matplotlib')

plot3 = plt.show()