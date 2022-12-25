import numpy as np
z=input("enter the value of z:")
def sigmoid_function():
    v=np.exp(-float(z))
    a= 1/(1+v)
    print(a)
def tanh():
    v1=np.exp(float(z))
    v2=np.exp(-float(z))
    b=(v1-v2)/(v1+v2)
    print(b)
def ReLU():
    if float(z)>0:
        print(z)
    else:
        print(0)
sigmoid_function()
tanh()
ReLU()