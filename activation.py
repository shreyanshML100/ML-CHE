import numpy as np
def tanh(z):
    return (np.e**z-np.e**(-1*z))/(np.e**z+np.e**(-1*z))
def ReLU(z):
    if z>0:
        return z
    if z<=0:
        return 0
def s(z):
    return 1/(1+np.e**(-1*z))
z=input()
z=float(z)
print(tanh(z))
print(s(z))
print(ReLU(z))