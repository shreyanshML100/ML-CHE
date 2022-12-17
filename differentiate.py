import activation as act
def sigmoid(z):
    h= 0.00001
    a=act.Sigmoid(z-h);
    b=act.Sigmoid(z-2*h);
    c=act.Sigmoid(z+h);
    d=act.Sigmoid(z+2*h);
    dif=(b- 8*a + 8*c -d)/(12*h);
    return dif;

def tanh(z):
    h= 0.00001
    a=act.Tanh(z-h);
    b=act.Tanh(z-2*h);
    c=act.Tanh(z+h);
    d=act.Tanh(z+2*h);
    dif=(b - 8*a + 8*c - d)/(12*h);
    return dif;

def ReLU(z):
    if(z<=0): 
     h = 0.00001
     a=act.ReLU(z);
     b=act.ReLU(z-h);
     c=act.ReLU(z- 2*h);
     dif= (3*a - 4*b + c)/(2*h)
     return dif;
    else:
      h= 0.00001
      a=act.ReLU(z);
      b=act.ReLU(z+h);
      c=act.ReLU(z+ 2*h);
      dif= (-3*a + 4*b - c)/(2*h)
      return dif; 
