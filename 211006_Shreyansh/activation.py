import numpy as np
import math
def sigmoind(z):
    if type(z)==np.ndarray:
        y=np.empty_like(z,float)
        for i in range(0,len(z)):
            y[i]=1/(1+math.exp(-z[i]))
        return(y)
    else:
        return(1/(1+math.exp(-z)))

def tanh(z):
    if type(z)==np.ndarray:
        y=np.empty_like(z,float)
        for i in range(0,len(z)):
            y[i]=(math.exp(z[i])-math.exp(-z[i]))/(math.exp(z[i])+math.exp(-z[i]))
        return(y)
    else:
        return((math.exp(z)-math.exp(-z))/(math.exp(z)+math.exp(-z)))

def ReLU(z):
    if type(z)==np.ndarray:
        y=np.empty_like(z,float)
        for i in range(0,len(z)):
            y[i]=(z[i] if z[i]>0 else 0)
        return(y)
    else:
        return(z if z>0 else 0)

