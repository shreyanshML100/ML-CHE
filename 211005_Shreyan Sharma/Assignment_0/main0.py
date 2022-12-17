import numpy as np
import graph as gp
import activation as f1
z=np.array([-100,-15,-9,-5,-1,0,1,5,9,15,100],dtype=float);
r=f1.Sigmoid(z);
print(r)
z=np.array([-100,-15,-9,-5,-1,0,1,5,9,15,100],dtype=float);
r=f1.Tanh(z);
print(r)
z=np.array([-100,-15,-9,-5,-1,0,1,5,9,15,100]);
r=f1.ReLU(z);
print(r)
gp.sigmoid(-10,10)
gp.tanh(-10,10)
gp.ReLU(-10,10)

ans=input("To enter another real number, press Y\n")
if ans=='Y':
    a=int(input("Enter the number\n"));
    t=a;
    r=f1.Sigmoid(a);
    print(r)   
    a=t; 
    r=f1.Tanh(a);
    print(r)  
    a=t;  
    r=f1.ReLu(a);
    print(r);  

