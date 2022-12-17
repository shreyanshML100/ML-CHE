import activation
h=0.0001
def diff_sigmoid(z) :
    return  (activation.sigmoid(z+h)-activation.sigmoid(z-h))/(2*h)
def diff_tanh(z) :
    return  (activation.tanh(z+h)-activation.tanh(z-h))/(2*h)
def diff_ReLU(z) :
    return  (activation.ReLU(z+h)-activation.ReLU(z-h))/(2*h)
