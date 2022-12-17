import numpy as np
 import matplotlib.pyplot as plt
 from activation import *

 def sigmoid_g(a,b):
     x=np.linspace(a,b,100)
     y=sigmoid(x)
     plt.pyplot.plot(x,y,marker=".")
     plt.pyplot.xlabel('z')
     plt.pyplot.ylabel('Sigmoid(z)')
     plt.pyplot.title("z vs Sigmoid(z)")
     plt.pyplot.show()

 def tanh_g(a,b):
     x=np.linspace(a,b,100)
     y=tanh(x)
     plt.pyplot.plot(x,y,marker=".")
     plt.pyplot.xlabel('z')
     plt.pyplot.ylabel('Sigmoid(z)')
     plt.pyplot.title("z vs Sigmoid(z)")
     plt.pyplot.show()

 def relu_g(a,b):
     x=np.linspace(a,b,100)
     y=relu(x)
     plt.pyplot.plot(x,y,marker=".")
     plt.pyplot.xlabel('z')
     plt.pyplot.ylabel('Sigmoid(z)')
     plt.pyplot.title("z vs Sigmoid(z)")
     plt.pyplot.show()

