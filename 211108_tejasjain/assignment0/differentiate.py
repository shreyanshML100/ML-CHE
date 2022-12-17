import activation 
import numpy as np
import math
def sigmoind(z):
    if type(z)==np.ndarray:
        y=np.empty_like(z,float)
        for i in range(0,len(z)):
            input=np.arange(z[i]-2*0.01,z[i]+3*0.01,0.01)
            f=activation.sigmoind(input)
            y[i]=(-f[4]+8*f[3]-8*f[1]+f[0])/(12*0.01)
        return(y)
    else:
        input=np.arange(z-2*0.01,z+3*0.01,0.01)
        f=activation.sigmoind(input)
        diff=(-f[4]+8*f[3]-8*f[1]+f[0])/(12*0.01)
        return diff

def tanh(z):
    if type(z)==np.ndarray:
        y=np.empty_like(z,float)
        for i in range(0,len(z)):
            input=np.arange(z[i]-2*0.01,z[i]+3*0.01,0.01)
            f=activation.tanh(input)
            y[i]=(-f[4]+8*f[3]-8*f[1]+f[0])/(12*0.01)
        return(y)
    else:
        input=np.arange(z-2*0.01,z+3*0.01,0.01)
        f=activation.tanh(input)
        diff=(-f[4]+8*f[3]-8*f[1]+f[0])/(12*0.01)
        return diff

def ReLU(z):
    if type(z)==np.ndarray:
        y=np.empty_like(z,float)
        for i in range(0,len(z)):
            input=np.arange(z[i]-2*0.01,z[i]+3*0.01,0.01)
            f=activation.ReLU(input)
            y[i]=(-f[4]+8*f[3]-8*f[1]+f[0])/(12*0.01)
        return(y)
    else:
        input=np.arange(z-2*0.01,z+3*0.01,0.01)
        f=activation.ReLU(input)
        diff=(-f[4]+8*f[3]-8*f[1]+f[0])/(12*0.01)
        return diff
