import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import activation
import graph

if __name__=="__main__":
    z=np.array([-100,-15,-9,-5,-1,0,5,9,15,100])
    print("\nsigmoid(z)=\n",activation.sigmoid(z))
    print("\ntanh(z)=\n",activation.tanh(z))
    print("\nReLU(z)=\n",activation.ReLU(z))

    graph.sigmoid(-10,10)
    graph.tanh(-10,10)
    graph.ReLU(-10,10)
