import matplotlib.pyplot as plt
import numpy as np
import  activation as act

def sig_plot(a,b):
    x = np.arange(a,b);
    y = act.Sigmoid(x);
    plt.title("Sigmoid");
    plt.plot(x,y);
    plt.show();

def tanh_plot(a,b):
    x = np.arange(a,b);
    y = act.tanh(x);
    plt.title("Tanh");
    plt.plot(x,y);
    plt.show();
def relu_plot(a,b):
    x = np.arange(a,b);
    y = act.ReLU(x);
    plt.title("ReLU");
    plt.plot(x,y);
    plt.show();    
    