import activation as act
import graph
import numpy as np
print( act.sigmoid_function(np.array([-100, -15, -9, -1, 0, 1, 5, 9, 15, 100])))
print( act.tanh(np.array([-100, -15, -9, -1, 0, 1, 5, 9, 15, 100])))
print( act.ReLU(np.array([-100, -15, -9, -1, 0, 1, 5, 9, 15, 100])))
graph.sigmoid_function(-10,10)
graph.tanh(-10,10)
graph.ReLU(-10,10)

