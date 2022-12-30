import numpy as np


def Sigmoid(z):
    return 1 / (1 + np.exp(-z))


def Tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(-z) + np.exp(z))


def ReLU(z):
    return np.maximum(0, z)


# n = np.array([1, 2, -3, 4, -5, -98])
# m = ReLU(n)
# print(m)
