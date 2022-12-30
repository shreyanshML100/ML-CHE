import numpy as np

h=10^(-16)
x=0
def sigmoid(x):
    s=1/(1+np.exp(-x))
    ds=s*(1-s)  
    errorsigmoid=ds-(1/(1+np.exp(-(x+h)))-1/(1+np.exp(-x)))/2*h
    print(errorsigmoid)
    return ds

def tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    dt=1-t**2
    errortanh=dt-((np.exp(x+h)-np.exp(-x-h))/(np.exp(x+h)+np.exp(-x-h))-t)/2*h
    print(errortanh)
    return dt

def relu(x):
    if(x<0):
         dr=0
    else:
         dr=1
    return dr

print("calling: sigmoid")
print(sigmoid(x))
print("calling: tanh")
print(tanh(x))
print("calling: RELU")
print(relu(x))