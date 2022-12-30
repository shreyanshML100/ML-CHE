import activation as f1
def sigmoid(z):
    a=f1.Sigmoid(z-0.1);
    b=f1.Sigmoid(z-0.2);
    c=f1.Sigmoid(z+0.1);
    d=f1.Sigmoid(z+0.2);
    diff=(b-8*a+8*c-d)/(12*0.1);
    return diff;

def tanh(z):
    a=f1.Tanh(z-0.1);
    b=f1.Tanh(z-0.2);
    c=f1.Tanh(z+0.1);
    d=f1.Tanh(z+0.2);
    diff=(b-8*a+8*c-d)/(12*0.1);
    return diff;

def ReLU(z):
    if(z>0): return 1;
    if (z<0): return 0;
    else: return "Not Defined";

num=int(input("Enter a number\n"));
ans=sigmoid(num);
print(ans);
ans=tanh(num);
print(ans);
ans=ReLU(num);
print(ans);