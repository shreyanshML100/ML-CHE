import numpy as np
import matplotlib.pyplot as plt

a=int(input("a:"))
b=int(input("b:"))
z=int(input("z:"))

def sigmoid(z):
    return 1/(1+np.exp(-z))
z = np.linspace(a, b)
plt.plot(z, sigmoid(z))
plt.axis('tight')
plt.title('Activation Function :Sigmoid')
plt.show()

def tanh(z):
     return np.tanh(z)
z = np.linspace(a, b)
plt.plot(z, tanh(z))
plt.axis('tight')
plt.title('Activation Function :Tanh')
plt.show()

def ReLU(z):
    z1 = []
    for i in z:
        if i < 0:
            z1.append(0)
        else:
            z1.append(i)

    return z1
z = np.linspace(a, b)
plt.plot(z, tanh(z))
plt.axis('tight')
plt.title('Activation Function :ReLU')
plt.show()


