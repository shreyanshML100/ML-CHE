import differentiate as d
import activation 
import numpy as np
z=np.array([-100,-15,-9,-5,-1,0,1,5,9,15,100])
print("Derivative of Sigmoind Function:")
print(d.sigmoind(z))
print("Derivative of Sigmoind Function:")
print(d.tanh(z))
print("Derivative of Sigmoind Function:")
print(d.ReLU(z))
