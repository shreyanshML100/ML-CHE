import numpy as np
import matplotlib as plt
from activation import * 
from graph import *

def diff_sigmoid(z):
    h=0.000001
    k=sigmoid(z+h)-sigmoid(z-h)
    return k/0.000002

def diff_tanh(z):
    h=0.000001
    k=tanh(z+h)-tanh(z-h)
    return k/0.000002

def diff_relu(z):
    h=0.000001
    k=relu(z+h)-relu(z-h)
    return k/0.000002