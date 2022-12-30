import activation as at
import graph as gp
import numpy as np
print(at.sigm(np.array([-100, -15, -9, -5, -1, 0, 1, 5 , 9, 15, 100])))
print(at.tanh(np.array([-100, -15, -9, -5, -1, 0, 1, 5 , 9, 15, 100])))
print(at.ReLU(np.array([-100, -15, -9, -5, -1, 0, 1, 5 , 9, 15, 100])))

gp.sigmoid(-10,10)
gp.tanh(-10,10)
gp.ReLU(-10,10)
