import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as m
from graph import *
from activation import *

# 1st Question
# defining the numpy array
z = [-100, -15, -9, -5, -1, 0, 1, 5, 9, 15, 100]

print("Sigmoid function -> \u03C3(z) = ", Sig_array(z), "\nTanh function -> tanh(z) = ",
      Tanh_array(z), "\nReLU Function -> ReLU(z) = ", ReLU_array(z))


# 2nd Qustion
plot_Sig(-10, 10)
plot_Tanh(-10, 10)
plot_ReLU(-10, 10)
