from math import exp

def sigmoid(x):
    f = lambda x : 1 / (1 + exp(-x))
    h = 0.1
    df = (f(x+h) - f(x)) / h
    print(df)

def tanh(x):
    f = lambda x : (exp(z) - exp(-z))/(exp(z) + exp(-z))
    h = 0.1
    df = (f(x+h) - f(x)) / h
    print(df)

def ReLU(x):
    if x > 0:
        print(1)
    else:
        print(0)

ReLU(100)

