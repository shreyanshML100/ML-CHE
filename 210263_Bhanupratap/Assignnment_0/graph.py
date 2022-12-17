import numpy as np
import matplotlib.pyplot as plt
import activation as act

def sigmoidGraph(a,b,x):
    x = np.linspace(a,b,20)
    y1= act.sigmoid(x)
    plt.plot(x,y1)
    plt.title("sigmoid")
    plt.ylabel("sigmoid(z)")
    plt.xlabel("z")
    plt.show()

def tanhGraph(a,b,x):
    x = np.linspace(a,b,20)
    y2 = act.tanh(x)
    plt.plot(x,y2)
    plt.title("tanh")
    plt.ylabel("tanh(z)")
    plt.xlabel("z")
    plt.show()

def ReLUGraph(a,b,x):
    x = np.linspace(a,b,20)
    y3 = act.ReLU(x)
    plt.plot(x,y3)
    plt.title("ReLU")
    plt.ylabel("ReLU(z)")
    plt.xlabel("z")
    plt.show()


# z = np.array([-100, -15, -9, -5, -1, 0, 1, 5 , 9, 15, 100])
# sigmoidGraph(-11,11,z)