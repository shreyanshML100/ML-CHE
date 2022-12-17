import activation
import graph
import numpy as np
z=np.array([-100,-15,-9,-5,-1,0,1,5,9,15,100])
print("Sigmoind Function:")
print(activation.sigmoind(z))
print("Tanh Function:")
print(activation.tanh(z))
print("ReLU Function:")
print(activation.ReLU(z))
graph.sigmoind(-10,10)
graph.tanh(-10,10)
graph.ReLU(-10,10)

