import numpy as np
import matplotlib.pyplot as plt

def CostFun(theta, X_train, Y_train):
    X=[]
    for f in X_train:
        v=f.copy()
        v.append(1)
        X.append(v)
    sum=0
    Y_train=np.array(Y_train)
    theta=np.array(theta)
    for o in range(len(X)):
        k=np.array(X[o])
        a=theta*k
        b=np.sum(a)-Y_train[o]
        sum+=((b**2))
    return sum/(2*len(X))

def DiffCostFun(theta, X_train, Y_train):
    X=[]
    for f in X_train:
        v=f.copy()
        v.append(1)
        X.append(v)
    diffj=[]
    Y_train=np.array(Y_train)
    theta=np.array(theta)
    for j in range(len(X[0])):
        sum=0
        for o in range(len(X)):
            i=X[o]
            k=np.array(i)
            a=theta*k
            b=(np.sum(a)-Y_train[o])*i[j]
            sum=sum+b
        diffj.append(sum/len(X))
    diffj=np.array(diffj)
    return diffj

def FitGD(theta,X_train,Y_train,alpha,lambdaa,type,no_ofiter):
    m=len(X_train)
    atheta=np.array(theta)
    J=[]
    iter=[]

    if type==1:
        u=1
        while u<=no_ofiter:
            p=np.sum(abs(atheta))
            p1=(lambdaa/m)*p
            err=CostFun(theta,X_train,Y_train)+p1
            J.append(err)
            iter.append(u)
            q=(lambdaa/m)
            grad=DiffCostFun(theta, X_train, Y_train)+q
            theta=theta-((alpha)*grad)
            u+=1

    if type==2:
        u=1
        while u<=no_ofiter:
            p=np.sum((atheta)**2)
            p1=(lambdaa/(2*m))*p
            err=CostFun(theta,X_train,Y_train)+p1
            J.append(err)
            iter.append(u)
            q=(lambdaa/m)*(atheta)
            grad=DiffCostFun(theta, X_train, Y_train)+q
            theta=theta-((alpha)*grad)
            u+=1

    if type==3:
        u=1
        while u<=no_ofiter:
            s=np.sum(abs(atheta))
            s1=(lambdaa/m)*s
            p=np.sum((atheta)**2)
            p1=(lambdaa/(2*m))*p
            err=CostFun(theta,X_train,Y_train)+p1+s1
            J.append(err)
            iter.append(u)
            q=(lambdaa/m)*(atheta)
            q1=(lambdaa/m)
            grad=DiffCostFun(theta, X_train, Y_train)+q1+q
            theta=theta-((alpha)*grad)
            u+=1

    '''plt.plot(iter,J)
    plt.ylabel("Cost Function ( J )")
    plt.xlabel("No. of Iterations")
    plt.show()'''
    return theta        

def FitNormal(X_train,Y_train):
    Y=np.array(Y_train)
    X=[]
    for f in X_train:
        v=f.copy()
        v.append(1)
        X.append(v)
    X=np.array(X)
    theta=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    return theta

def LocallyWeighted(X_train,Y_train,x,tau,no_ofiter):
    x.append(1)
    x=np.array(x)
    X=[]
    for f in X_train:
        v=f.copy()
        v.append(1)
        X.append(v)
    X=np.array(X)
    Y=np.array(Y_train)
    m=len(X)
    n=len(X[0])
    W=[1]*m
    u=0
    while(u<=no_ofiter):
        for i in range (m) :
            xi=np.array(X[i])
            cst=(-1)/(2*(tau**2))
            k=np.sum(abs(xi-x))
            W[i]=np.exp(cst*k)
            u+=1
    W=np.diag(W)
    theta=np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(Y)
    y=theta*x
    return np.sum(y)