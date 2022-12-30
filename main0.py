import activation as av
import graph as gr
import numpy as np

z=np.array([-100,-15,-9,-5,-1,0,1,5,9,15,100])
print(av.sigmoid(z))
print(av.tanh(z))
print(av.ReLU(z))

l=-10
u=10

gr.sigma_plot(float(l),float(u))
gr.tanh_plot(float(l),float(u))
gr.ReLU_plot(float(l),float(u))