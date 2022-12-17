import numpy as np
from activation import *
from graph import *

# (a)
print("1. III (a)")
z = np.array([-100, -15, -9, -5, -1, 0, 1, 5 , 9, 15, 100])

#sigmoid
print("sigmoid")
for i in range(0,len(z)):
    print(sigmoid(z[i]))

#tanh
print("tanh")
for i in range(0,len(z)):
    print(tanh(z[i]))

#ReLU
print("ReLU")
for i in range(0,len(z)):
    print(ReLU(z[i]))

print("\n")


#(b) 
print("1. III (b)")
a = -10
b = 10

#sigmoid
print("sigmoid graph")
print(sigmoidG(a, b))

print("tanh graph")
print(sigmoidG(a, b))

print("ReLU graph")
print(ReLUG(a, b))


