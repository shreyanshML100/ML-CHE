import numpy as np
import activations
import graph

z = np.array([-100, -15, -9, -5, -1, 0, 1, 5 , 9, 15, 100])

print(activations.sigmoid(z))
print()
print(activations.tanh(z))
print()
print(activations.ReLU(z))
print()

graph.sigmoid(-10,10)
graph.tanh(-10,10)
graph.ReLU(-10,10)