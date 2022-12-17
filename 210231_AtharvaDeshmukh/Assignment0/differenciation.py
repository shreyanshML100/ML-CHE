import activation as activation
h=0.0001
m=2*h
def sigmoid(z) :
    return  (activation.sigmoid(z+h)-activation.sigmoid(z-h))/m
def tanh(z) :
    return  (activation.tanh(z+h)-activation.tanh(z-h))/m
def ReLU(z) :
    return  (activation.ReLU(z+h)-activation.ReLU(z-h))/m