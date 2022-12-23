import numpy as np
import matplotlib.pyplot as plt
import graph
import activation as act

## First part ---->
z = np.array([-100, -15, -9, -5, -1, 0, 1, 5 , 9, 15, 100])

print("sigmoid Values: ", act.sigmoid(z))
print("tanh Values: ",act.tanh(z))
print("reLU Values: ", act.ReLU(z))



## Second part ---->
graph.sigmoidGraph(-10,10,z)
graph.tanhGraph(-10,10,z)
graph.ReLUGraph(-10,10,z)
