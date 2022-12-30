import activation as act
import matplotlib.pyplot as plt
import numpy as np




def sigmoid(a,b):
    z = np.linspace(a,b,num=1000)
    y = act.sigmoid(z)
    plt.plot(z,y,label="sigmoid")

def tanh(a,b):
    z = np.linspace(a, b, num=1000)
    y = act.tanh(z)
    plt.plot(z, y,label="tanh")

def ReLU(a,b):
    z = np.linspace(a, b, num=1000)
    y = act.ReLU(z)
    plt.plot(z, y,label="ReLU")
    plt.legend()


plt.show()