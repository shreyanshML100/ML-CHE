import numpy as np
import matplotlib as plt

def sigmoid(z):
    k=np.exp(z)
    p=1+(1/k)
    return (1/p)

def tanh(z):
    a=np.exp(z)
    b=1/a
    c=(a-b)/(a+b)
    return c

def relu(z):
    n=len(z)
    for i in range(n):
        if z[i]<=0:
            z[i]=0
    return z