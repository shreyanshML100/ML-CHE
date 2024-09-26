import math
import numpy as np
import matplotlib.pyplot as plt
import activation

def sigmoid(a,b):
    
    n=np.arange(a,b,(b-a)/100.)
    plt.title("Sigmoid")
    plt.plot(n,activation.sigmoid(n))
    plt.show()
    return 
def tanh(a,b):
    n=np.arange(a,b,(b-a)/100.)
    plt.title("Tanh")
    plt.plot(n,activation.tanh(n))
    plt.show()
    return 
def relu(a,b):
    n=np.arange(a,b,(b-a)/100.)
    plt.title("ReLU")
    plt.plot(n,activation.relu(n),'g-')
    plt.show()
    return 
