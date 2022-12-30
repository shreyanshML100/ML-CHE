import activation as av
import graph as gr
import numpy as np

def sigma_diff(a):
        return (av.sigmoid(a+0.001)-av.sigmoid(a-0.001))/(0.002)
   
def tanh_diff(a):
        return (av.tanh(a+0.001)-av.tanh(a-0.001))/(0.002)
    
def ReLU_diff(a):
        return (av.ReLU(a+0.001)-av.ReLU(a-0.001))/(0.002)
   