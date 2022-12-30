import numpy as np
import matplotlib.pyplot as plt

z = np.array([ -100, -15, -9, -5, -1, 0, 1, 5 , 9, 15, 100, 0])

from activation import sigmoid
from activation import tanh
from activation import ReLU

print(sigmoid(z))
print(tanh(z))
ReLU.all()
print(ReLU(z))

from graph import sigmoid
from graph import tanh
from graph import ReLU








