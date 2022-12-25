import numpy as np
import scipy as sp
import sympy as smp
from scipy.misc import derivative
def sigmoid_function():
    z = smp.symbols('z', real=True)
    f=1/(1+smp.exp(-z))
    dfdz= smp.diff(f,z)
    print(dfdz)
sigmoid_function()
def tanh():
    z = smp.symbols('z', real=True)
    f=(smp.exp(z)-smp.exp(-z))/(smp.exp(z)+smp.exp(-z))
    dfdz=smp.diff(f,z)
    print(dfdz)
tanh()
def ReLU():
    z=input("enter the value of z:")
    if int(z)>0:
        f=z
        print(1)
    else:
        f=0
        print(0)
ReLU()