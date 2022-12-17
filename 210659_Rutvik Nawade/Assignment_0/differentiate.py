import numpy as np
import activation


def diff_Sigmoid(z):
    z1 = activation.Sigmoid(z) * (1 - activation.Sigmoid(z))
    return z1


def diff_Tanh(z):
    z2 = 1 - activation.Tanh(z) ** 2
    return z2


def diff_ReLU(z):
    return np.maximum(0, z / np.absolute(z))


# n = np.array([1, 2, -3, 4, -5, -98])
# m1 = diff_Sigmoid(n)
# print(m1)