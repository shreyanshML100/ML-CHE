import numpy as np


def sigmoid(array):
    x1=np.array([])
    for x in array:
        x1=np.append(x1,[1/(1+np.exp(-x))])
    print(x1)
    return x1

def tanh(array):
    x2=np.array([])
    for x in array:
        x2=np.append(x2,[np.tanh(x)])
    print(x2)
    return x2


def RELU(array):
    x3=np.array([])
    for i in array:
        if i<0:
            x3=np.append(x3,[0])
        else:
            x3=np.append(x3,[i])
    print(x3)
    return x3


  



