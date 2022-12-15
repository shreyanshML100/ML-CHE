import numpy as np

import activation as act
import graph

#z = [-100, -15, -9, -5, -1, 0, 1, 5 , 9, 15, 100]
z=np.array([-100, -15, -9, -5, -1, 0, 1, 5 , 9, 15, 100])
print("Activation Values for sigmoid function "+str(act.sigmoid(z)))
print("Activation Values for ReLU function "+str(act.ReLU(z)))
print("Activation Values for tanh function "+str(act.tanh(z)))

graph.sigmoid(-10,10)
graph.tanh(-10,10)
graph.ReLU(-10,10)
