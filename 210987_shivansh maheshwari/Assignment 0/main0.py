import numpy as np
import activation
import graph
x = np.array([-100, -15, -9, -5, -1, 0, 1, 5 , 9, 15, 100])
print("Sigmoid:",activation.sigmoid(x))
print("Tanh:",activation.tanh(x))
print("ReLU:",activation.relu(x))
graph.sigmoid(-10,10)
graph.tanh(-10,10)
graph.relu(-10,10)

