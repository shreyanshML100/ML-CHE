import math
import numpy as np
import matplotlib.pyplot as plt
from activation import *


def sigmoidGraph(a, b):
    x = np.arange(a, b, 1.0000e-05)
    y = np.empty(len(x))
    y = sigmoid(x)
    plt.plot(x, y)


def tanhGraph(a, b):
    x = np.arange(a, b, 1.0000e-05)
    y = np.empty(len(x))
    y = tanh(x)
    plt.plot(x, y)


def reluGraph(a, b):
    x = np.arange(a, b + 1)
    y = np.empty(len(x))
    y = relu(x)
    plt.plot(x, y)
