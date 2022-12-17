from Activation import *
from Graph import plot1,plot2,plot3
import numpy as np

z = np.array([-100, -15, -9, -5, -1, 0, 1, 5 , 9, 15, 100])

print('sigmoid = ',sigmoid(z))
print()
print('tanh = ',tanh([z]))
print()
print('ReLU = ',ReLU(z))
