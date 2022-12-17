import numpy as np


def sigmoind(z):
    if type(z)==np.ndarray:
        y=np.empty_like(z,float)
        for i in range(0,len(z)):
            y[i]=1/(1+np.exp(-z[i]))
        return(y)
    else:
        return(1/(1+np.exp(-z)))

def tanh(z):
    if type(z)==np.ndarray:
        y=np.empty_like(z,float)
        for i in range(0,len(z)):
            y[i]=(np.exp(z[i])-np.exp(-z[i]))/(np.exp(z[i])+np.exp(-z[i]))
        return(y)
    else:
        return((np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z)))

def ReLU(z):
    if type(z)==np.ndarray:
        y=np.empty_like(z,float)
        for i in range(0,len(z)):
            y[i]=(z[i] if z[i]>0 else 0)
        return(y)
    else:
        return(z if z>0 else 0)

