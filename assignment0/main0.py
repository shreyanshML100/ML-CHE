import activation as act
import graph as graph
import numpy as np

a = np.array([ -100, -15, -9, -5, -1, 0, 1, 5 , 9, 15, 100, 0])

print('Activation is')
print(act.sigmoid(a))
print(act.tanh(a))
print(act.ReLU(a))

print('graph is')
graph.sigmoid(-10,10)
graph.tanh(-10,10)
graph.ReLU(-10,10)
