# Sigmoid Function
import math as m
import numpy as np
from numpy import *

# Sigmoid Function


def Sig(z):
    F = 1/(1+pow(m.e, -1*z))
    return F
# Tanh Function


def Tanh(z):
    F = (pow(m.e, z)-pow(m.e, -1*z))/(pow(m.e, z)+pow(m.e, -1*z))
    return F
# ReLU Function


def ReLU(z):
    if z > 0:
        return float(z)
    elif z <= 0:
        return float(0)

# Activation functions for arrays


def Sig_array(z):
    func1 = np.vectorize(Sig)
    return (func1(z))


def Tanh_array(z):
    func2 = np.vectorize(Tanh)
    return (func2(z))


def ReLU_array(z):
    func3 = np.vectorize(ReLU)
    return (func3(z))
