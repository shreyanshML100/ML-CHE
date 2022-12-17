import numpy as np
import matplotlib.pyplot as plt
from activation import *
from graph import *

a = [-100, -15, -9, -5, -1, 0, 1, 5 , 9, 15, 100]
z=np.array(a)

sigmoid(z)
tanh(z)
relu(z)

sigmoid_g(-10,10)
tanh_g(-10,10)
relu_g(-10,10)