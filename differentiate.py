import numpy as np

def tanh(z):
    return ((np.e**(z+0.05)-np.e**(-1*z-0.05))/(np.e**(z+0.05)+np.e**(-1*z-0.05))-(np.e**(z-0.05)-np.e**(-1*z+0.05))/(np.e**(z-0.05)+np.e**(-1*z+0.05)))/0.1
def ReLU(z):
    if z>0:
        return 1
    if z<0:
        return 0
def s(z):
    return (1/(1+np.e**(-1*z-0.05))-1/(1+np.e**(-1*z+0.05)))/0.1

z=input()
z=float(z)
print(tanh(z))
print(s(z))
print(ReLU(z))