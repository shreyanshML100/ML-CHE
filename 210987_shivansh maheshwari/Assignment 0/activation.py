import numpy as np
import math

def sigmoid(n):
    
       
    n=1/(1+math.e**(-n))
    return n
def tanh(n):
    
    n= (math.e**(n)-math.e**(-n))/(math.e**(n)+math.e**(-n))
    
    return n
def relu(n):
    if type(n)==int or type(n)==float:
        if n<=0:
            return 0.
        else :
            return n
    else :
        m=n.shape
        for i in range(m[0]):
            if n[i]<0 or n[i]==0:
                n[i]=0.
        return n
            


