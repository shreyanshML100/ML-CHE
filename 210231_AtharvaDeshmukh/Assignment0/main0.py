import activation as activation
import differenciation as differenciation
import graph as graph
import numpy as np

arr = np.array([ -100, -15, -9, -5, -1, 0, 1, 5 , 9, 15, 100, 0])

print('Activation -')
print(activation.sigmoid(arr))
print(activation.tanh(arr))
print(activation.ReLU(arr))

graph.sigmoid(-10,10)
graph.tanh(-10,10)
graph.ReLU(-10,10)

print('Differenciation -')
print(differenciation.sigmoid(arr))
print(differenciation.tanh(arr))
print(differenciation.ReLU(arr))