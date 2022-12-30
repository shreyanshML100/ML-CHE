import activation as av
import matplotlib.pyplot as plt
import numpy as np

def sigma_plot(a,b):
    x_coord = np.linspace(a,b,(int(b-a))*10)
    y_coord = av.sigmoid(x_coord)
    plt.plot(x_coord,y_coord)
    plt.show()

def tanh_plot(a,b):
    x_coord = np.linspace(a,b,(int(b-a))*10)
    y_coord = av.tanh(x_coord)
    plt.plot(x_coord,y_coord)
    plt.show()

def ReLU_plot(a,b):
    x_coord = np.linspace(a,b,(int(b-a))*10)
    y_coord = av.ReLU(x_coord)
    plt.plot(x_coord,y_coord)
    plt.show()

