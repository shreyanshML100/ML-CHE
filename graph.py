import matplotlib.pyplot as plt
from activation import *
import numpy as np

input=np.linspace(-10,10)
plt.plot(input,1/(1 + np.exp(-input)))
plt.xlabel("x")
plt.ylabel("Sigmoid(X)")
plt.title('Activation Function:Sigmoid')
plt.show()

x=input
plt.plot(input,tanh(input))
plt.xlabel("x")
plt.ylabel("tanh")
plt.title('Activation Function:tanh')
plt.show()

    
plt.plot(input,RELU(input))
plt.xlabel("x")
plt.ylabel("RELU")
plt.title('Activation Function:RELU')
plt.show()