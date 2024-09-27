import activation as act
def Sigmoid(z):
    h= 0.00001
    a=act.sigmoid(z-h);
    b=act.sigmoid(z-2*h);
    c=act.sigmoid(z+h);
    d=act.sigmoid(z+2*h);
    dif=(b- 8*a + 8*c -d)/(12*h);
    return dif;

def Tanh(z):
    h= 0.00001
    a=act.tanh(z-h);
    b=act.tanh(z-2*h);
    c=act.tanh(z+h);
    d=act.tanh(z+2*h);
    dif=(b - 8*a + 8*c - d)/(12*h);
    return dif;

def reLU(z):
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
