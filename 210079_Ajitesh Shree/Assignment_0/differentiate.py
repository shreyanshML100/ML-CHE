import numpy as np
from activation import *


def sigmoidDiff(array):
    return sigmoid(array) - sigmoid(array) * sigmoid(array)


def tanhDiff(array):
    return 1 - tanh(array) * tanh(array)


def reluDiff(array):
    answer = np.empty(len(array))
    i = 0
    for element in array:
        if element <= 0:
            answer[i] = 0
        else:
            answer[i] = 1
        i = i + 1
    return answer
