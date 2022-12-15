import numpy as np
import matplotlib.pyplot as plt
import graph
import activations as act

z = np.array([-100, -15, -9, -5, -1, 0, 1, 5 , 9, 15, 100])

print("Sigmoid Values", act.sigmoid(z))
print("Tanh Values",act.tanh(z))
print("ReLU Values", act.Relu(z))

graph.sig_graph(-10,10,z)
graph.tanh_graph(-10,10,z)
graph.ReLU_graph(-10,10,z)


