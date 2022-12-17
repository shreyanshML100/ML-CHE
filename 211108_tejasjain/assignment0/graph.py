import matplotlib.pyplot as mp
import numpy as np
import activation
def sigmoind(a,b):
    z=np.arange(a,b)
    y=activation.sigmoind(z)
    mp.figure(1)
    mp.plot(z,y)
    mp.grid()
    mp.xlabel("x")
    mp.ylabel("f(x)")
    mp.title("Sigmoind Function")
    mp.show()

def tanh(a,b):
    z=np.arange(a,b)
    y=activation.tanh(z)
    mp.figure(2)
    mp.plot(z,y)
    mp.grid()
    mp.xlabel("x")
    mp.ylabel("f(x)")
    mp.title("tanh Function")
    mp.show()
def ReLU(a,b):
    z=np.arange(a,b)
    y=activation.ReLU(z)
    mp.figure(3)
    mp.plot(z,y)
    mp.grid()
    mp.xlabel("x")
    mp.ylabel("f(x)")
    mp.title("ReLU Function")
    mp.show()
