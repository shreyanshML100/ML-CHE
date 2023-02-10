import numpy as np
import graph
import matplotlib.pyplot as plt
import activation as act

z = np.array([-100, -15, -9, -5, -1, 0, 1, 5 , 9, 15, 100])

print("Activation values for sigmoid", act.sigmoid(z))
print("Activation values for tanh", act.tanh(z))
print("Activation values for ReLU", act.ReLU(z))

graph.sigmoid(-10,10,z)
graph.tanh(-10,10,z)
graph.ReLU(-10,10,z)