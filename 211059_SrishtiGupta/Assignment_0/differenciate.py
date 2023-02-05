import activations

def sigmoid(z):
	h = 0.000000001
	x = (activations.sigmoid(z+h) - activations.sigmoid(z-h))/(2*h)
	return x

def tanh(z):
	h = 0.000000001
	x = (activations.tanh(z+h) - activations.tanh(z-h))/(2*h)
	return x

def ReLU(z):
	h = 0.000000001
	x = (activations.ReLU(z+h) - activations.ReLU(z-h))/(2*h)
	return x
