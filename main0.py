from activation import * 
import numpy as np
import graph as g


my_array=[-100,-15,-9,-5,-1,0,5,9,15,100]
my_array = np.array(my_array)
print("Activation: sigmoid")
sigmoid(my_array)
print(" ")
print("Activation: tanh")
tanh(my_array)
print("")
print("Activation: RELU")
RELU(my_array)
print("")

import graph