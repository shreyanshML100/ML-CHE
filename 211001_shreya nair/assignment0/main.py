import numpy as np
import activation as act
import graph

z = np.array([-100, -15, -9, -5, -1, 0, 1, 5, 9, 15, 100])
print(" Activation values for Sigmoid function " + str(act.Sigmoid(z)), "/n")
print(" Activation values for ReLU function " + str(act.ReLU(z)), "/n")
print(" Activation values for Tanh function " + str(act.Tanh(z)), "/n")

graph.Sigmoid(-10, 10)
graph.Tanh(-10, 10)
graph.ReLU(-10, 10)
