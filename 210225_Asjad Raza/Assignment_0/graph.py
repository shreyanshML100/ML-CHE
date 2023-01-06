import numpy as np
import matplotlib.pyplot as plt
import activations as act

def sig_graph(a,b,x):
    x = np.linspace(a,b)
    y1= act.sigmoid(x)
    plt.plot(x,y1)
    plt.title("Sigmoid")
    plt.show()
    
def tanh_graph(a,b,x):
    x = np.linspace(a,b)
    y2 = act.tanh(x)
    plt.plot(x,y2)
    plt.title("tanh")
    plt.show()

def ReLU_graph(a,b,x):
    x = np.linspace(a,b)
    y3 = act.Relu(x)
    plt.plot(x,y3)
    plt.title("ReLU")
    plt.show()
