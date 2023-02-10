from math import exp

def sigmoid(z):
    a = 1 / (1 + exp(-z))
    return a

def tanh(z):
    a = (exp(z) - exp(-z))/(exp(z) + exp(-z))
    return a
    
def ReLU(z):
    
    if z > 0:
        return z
    else:
        return 0
          





