import numpy as np
import activation
import graph

z = np.array([-100, -15, -9, -5, -1, 0, 1, 5, 9, 15, 100])
print(" Values for Sigmoid function " + str(activation.Sigmoid(z)), "/n")
print(" Values for ReLU function " + str(activation.ReLU(z)), "/n")
print(" Values for Tanh function " + str(activation.Tanh(z)), "/n")

graph.plt_Sigmoid(-10, 10)
graph.plt_Tanh(-10, 10)
graph.plt_ReLU(-10, 10)
