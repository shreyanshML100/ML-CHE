import numpy as np
import matplotlib.pyplot as plt
from activation import *
from graph import *

# a
z = np.array([-100, -15, -9, -5, -1, 0, 1, 5, 9, 15, 100])
print(sigmoid(z))
print()
print(tanh(z))
print()
print(relu(z))

# b
plt.subplot(2, 2, 1)
sigmoidGraph(-10, 10)
plt.subplot(2, 2, 2)
tanhGraph(-10, 10)
plt.subplot(2, 2, 3)
reluGraph(-10, 10)
plt.show()
