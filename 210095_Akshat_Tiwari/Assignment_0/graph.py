import matplotlib.pyplot as plt
import math as m
from activation import *


def plot_Sig(a, b):
    f = plt.figure(1)
    x1 = range(a, b)
    y1 = [Sig(x) for x in x1]
    plt.plot(x1, y1)
    plt.plot(x1, y1)
    plt.title('Plot of Sigmoid Function')
    plt.xlabel('This is the x-axis')
    plt.ylabel('This is the y-axis')
    f.show()


def plot_Tanh(a, b):
    g = plt.figure(2)
    x2 = range(a, b)
    y2 = [Tanh(x) for x in x2]
    plt.plot(x2, y2)
    plt.plot(x2, y2)

    plt.title('Plot of Tanh Function')
    plt.xlabel('This is the x-axis')
    plt.ylabel('This is the y-axis')
    g.show()


def plot_ReLU(a, b):
    h = plt.figure(3)
    x3 = range(a, b)
    y3 = [ReLU(x) for x in x3]
    plt.plot(x3, y3)
    plt.plot(x3, y3)

    plt.title('Plot of ReLU Function')
    plt.xlabel('This is the x-axis')
    plt.ylabel('This is the y-axis')
    plt.show()
