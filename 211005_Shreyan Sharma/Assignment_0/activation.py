


import math
import numpy as np
def Sigmoid(z):
            z=1/(1+np.exp(-z));

            return z;

def Tanh(z):
    
    z=(np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z));
        
    return z;  

def ReLU(z):
    for i in range(0,z.size):
        if(z[i]<=0): 
            z[i]=0;
        
    return z;

def ReLu(z):
    if(z<=0): 
        z=0;
        return z;
    else:
     return z;










