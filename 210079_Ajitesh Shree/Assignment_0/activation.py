import math
import numpy as np


def sigmoid(array):
    answer = np.empty(len(array))
    i = 0
    for element in array:
        answer[i] = 1 / (1 + math.exp(-element))
        i = i + 1
    return answer


def tanh(array):
    answer = np.empty(len(array))
    i = 0
    for element in array:
        answer[i] = (math.exp(element) - math.exp(-element)) / (
            math.exp(element) + math.exp(-element)
        )
        i = i + 1
    return answer


def relu(array):
    answer = np.empty(len(array))
    i = 0
    for element in array:
        if element <= 0:
            answer[i] = 0
        else:
            answer[i] = element
        i = i + 1
    return answer
