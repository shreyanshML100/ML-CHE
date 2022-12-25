import matplotlib as plt
import numpy as np
import activation as act

def diff_sig(z,h):
    return (act.Sigmoid(z+h)-act.Sigmoid(z-h))/2*h;
def diff_tanh(z,h):
    return (act.tanh(z+h)-act.tanh(z-h))/2*h;
def diff_relu(z,h):
    return (act.ReLU(z+h)-act.ReLU(z-h))/2*h;