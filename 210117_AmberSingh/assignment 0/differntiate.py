import numpy as np 
import matplotlib.pyplot as plt
## Central Difference is used for approximating derivatives...
def f(x):
    return (1/(1+np.exp(-x)))
def actualDerivative_1(x):
    return np.exp(-x)/((1+np.exp(-x))**2)
def sigmoid(f,x0,h):
    cd=(f(x0+h)-f(x0-h))/(2*h)
    return cd
print("Approximated Derivative is ",actualDerivative_1(0))
print("Actual Derivative is ",sigmoid(f,0,0.1))
print("error is",abs(actualDerivative_1(0)-sigmoid(f,0,0.1)),"\n")

def g(x):
    z=np.exp(-x)
    s=np.exp(x)
    y=(s-z)/(s+z)
    return y
def actualDerivative_2(x):
    return (1-g(x)**2)

def tanh(g,x0,h):
    cd=(g(x0+h)-g(x0-h))/(2*h)
    return cd
print("Approximated Derivative is ",actualDerivative_2(0))
print("Actual Derivative is ",tanh(g,0,0.1))
print("error is",abs(actualDerivative_2(0)-tanh(g,0,0.1)),"\n")


def p(x):
    return np.maximum(x,0)
def ReLU(p,x0,h):
    cd=(p(x0+h)-p(x0-h))/(2*h)
    return cd
def actualDerivative_3(x):
    if(x<0):
     return 0
    else:
     return 1
print("Approximated Derivative is ",ReLU(p,5,0.1))
print("Actual Derivative is ",actualDerivative_3(5))
print("error is",abs(actualDerivative_3(0)-ReLU(p,5,0.1)),"\n")
