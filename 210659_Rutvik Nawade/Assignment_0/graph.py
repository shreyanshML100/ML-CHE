import activation
import matplotlib.pyplot as plt
import numpy as np

# a = -5
# b = 5


def plt_Sigmoid(a, b):
    z = np.linspace(a, b, num=10000)
    y = activation.Sigmoid(z)
    plt.plot(z, y, label="Sigmoid")


def plt_Tanh(a, b):
    z = np.linspace(a, b, num=10000)
    y = activation.Tanh(z)
    plt.plot(z, y, label="Tanh")


def plt_ReLU(a, b):
    z = np.linspace(a, b, num=100)
    y = activation.ReLU(z)
    plt.plot(z, y, label="ReLU")
    plt.legend()


plt.show(plt_ReLU(-5, 5))
