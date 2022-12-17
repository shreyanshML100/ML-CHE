import activation
def sigmoid(a):
    return (activation.sigmoid(a+a/1000000)-activation.sigmoid(a-a/1000000))/(2*a/1000000)
def tanh(a):
    return (activation.tanh(a+a/1000000)-activation.tanh(a-a/1000000))/(2*a/1000000)
def relu(a):
    if(a>0) :
        return 1.
    return 0.