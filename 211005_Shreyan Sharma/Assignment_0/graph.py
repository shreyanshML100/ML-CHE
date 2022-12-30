import matplotlib.pyplot as plt
import numpy as np
import math
# plt.plot([1,2,3],[2,3,5])
# plt.show()

        
def sigmoid(a,b):
    z=np.linspace(a,b,1000)
    y=1/(1+math.exp(1)**(-z));
    plt.plot(z,y);

    plt.title("sigmoid function");
    plt.xlabel("z")
    plt.ylabel("sigmoid")
    plt.show()

def tanh(a,b):
    z=np.linspace(a,b,1000)
    y=(math.exp(1)**(z)-math.exp(1)**(-z))/(math.exp(1)**(-z)+math.exp(1)**(z));
    plt.plot(z,y);

    plt.title("tanh function");
    plt.xlabel("z")
    plt.ylabel("tanh")
    plt.show()

def ReLU(a,b):
    z=np.linspace(a,b,1000)
    if a>0:
        plt.plot(z,z);
    else:
        z1=np.linspace(0,b,1000);
        y=z1;
        plt.plot(z1,y,color="blue");
        z2=np.linspace(a,0,1000);
        y=0*z;
        plt.plot(z2,y,color="blue");

    plt.title("ReLU function");
    plt.xlabel("z")
    plt.ylabel("ReLU")
    plt.show()  



