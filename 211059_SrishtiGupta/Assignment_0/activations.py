import numpy as np

sigmoid = lambda z : 1/(1+(np.e)**(-z))
tanh = lambda z : (np.e**z - np.e**(-z))/(np.e**z + np.e**(-z))
def ReLU(z) :
    ct = 0
    if (np.size(z) != 1) :
        for x in z:
            if (x < 0):
                z[ct] = 0
            ct = ct+1
        return z
    else:
        if (z>0):
            return z
        else:
            return 0