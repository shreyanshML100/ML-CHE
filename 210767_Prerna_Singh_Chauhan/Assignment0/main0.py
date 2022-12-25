import matplotlib as plt
import numpy as np
import graph as gh
import activation as act

z = np.array([-100, -15, -9, -5, -1, 0, 1, 5 , 9, 15, 100]);

print("Sigmoid functions values: ",act.Sigmoid(z));
print("tanh function values: ",act.tanh(z));
print("ReLU function values : ",act.ReLU(z));

a = -10;
b = 10;

gh.sig_plot(a,b);
gh.tanh_plot(a,b);
gh.relu_plot(a,b);

