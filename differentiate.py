import activation as act
h=0.001
def sigmoid(z) :
    return  (act.sigmoid(z+h)-act.sigmoid(z-h))/(2*h)
def tanh(z) :
    return  (act.tanh(z+h)-act.tanh(z-h))/(2*h)
def ReLU(z) :
    return  (act.ReLU(z+h)-act.ReLU(z-h))/(2*h)

