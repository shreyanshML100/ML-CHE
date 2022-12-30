import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as m
from activation import *
from graph import *
h = 0.0001
m = 2*h


def deri_Sig(z):
    derivative = (Sig(z+h)-Sig(z-h))/m
    return (derivative)


def deri_Tanh(z):
    derivative = (Tanh(z+h)-Tanh(z-h))/m
    return (derivative)


def deri_ReLU(z):
    derivative = (ReLU(z+h)-ReLU(z-h))/m
    return (derivative)


def deri_Tanh_array(z):
    derivative = (Tanh_array(z+h)-Tanh_array(z-h))/m
    return (derivative)


def deri_Sig_array(z):
    derivative = (Sig_array(z+h)-Sig_array(z-h))/m
    return (derivative)


def deri_ReLU_array(z):
    derivative = (ReLU_array(z+h)-ReLU_array(z-h))/m
    return (derivative)
