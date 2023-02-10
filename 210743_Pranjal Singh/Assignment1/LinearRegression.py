import numpy as np
import matplotlib.pyplot as plt

def CostFun(theta, X_train, Y_train):
    X=[]
    for f in X_train:
        v=f.copy()
        v=np.append(v,1)
        X.append(v)
    ans=0
    Y_train=np.array(Y_train)
    theta=np.array(theta)
    for o in range(len(X)):
        k=np.array(X[o])
        a=theta*k
        b=np.sum(a)-Y_train[o]
        ans+=((b**2))
    return ans/(2*len(X))

def DiffCostFun(theta, X_train, Y_train):
    X=[]
    for f in X_train:
        v=f.copy()
        v=np.append(v,1)
        X.append(v)
    diffj=[]
    Y_train=np.array(Y_train)
    theta=np.array(theta)
    for j in range(len(X[0])):
        ans=0
        for o in range(len(X)):
            i=X[o]
            k=np.array(i)
            a=theta*k
            b=(np.sum(a)-Y_train[o])*i[j]
            ans=ans+b
        diffj.append(ans/len(X))
    diffj=np.array(diffj)
    return diffj

def FitGD(theta,X_train,Y_train,alpha,rate_of_regu,type_of_regu,iterations):
    m=len(X_train)
    atheta=np.array(theta)
    J=[]
    iter=[]

    if type_of_regu==1:
        u=1
        while u<=iterations:
            p=np.sum(abs(atheta))
            p1=(rate_of_regu/m)*p
            err=CostFun(theta,X_train,Y_train)+p1
            J.append(err)
            iter.append(u)
            q=(rate_of_regu/m)
            grad=DiffCostFun(theta, X_train, Y_train)+q
            theta=theta-((alpha)*grad)
            u+=1

    if type_of_regu==2:
        u=1
        while u<=iterations:
            p=np.sum((atheta)**2)
            p1=(rate_of_regu/(2*m))*p
            err=CostFun(theta,X_train,Y_train)+p1
            J.append(err)
            iter.append(u)
            q=(rate_of_regu/m)*(atheta)
            grad=DiffCostFun(theta, X_train, Y_train)+q
            theta=theta-((alpha)*grad)
            u+=1

    if type_of_regu==3:
        u=1
        while u<=iterations:
            s=np.sum(abs(atheta))
            s1=(rate_of_regu/m)*s
            p=np.sum((atheta)**2)
            p1=(rate_of_regu/(2*m))*p
            err=CostFun(theta,X_train,Y_train)+p1+s1
            J.append(err)
            iter.append(u)
            q=(rate_of_regu/m)*(atheta)
            q1=(rate_of_regu/m)
            grad=DiffCostFun(theta, X_train, Y_train)+q1+q
            theta=theta-((alpha)*grad)
            u+=1
    return theta        

def FitNormal(X_train,Y_train):
    Y=np.array(Y_train)
    X=[]
    for f in X_train:
        v=f.copy()
        v=np.append(v,1)
        X.append(v)
    X=np.array(X)
    theta=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    return theta

def LocallyWeighted(X_train,Y_train,x,tau,iterations):
    X=[]
    for f in X_train:
        v=f.copy()
        v=np.append(v,1)
        X.append(v)
    X=np.array(X)
    Y=np.array(Y_train)
    m=len(X)
    n=len(X[0])
    W=[1]*m
    u=0
    while(u<=iterations):
        for i in range (m) :
            xi=np.array(X[i])
            cst=(-1)/(2*(tau**2))
            k=np.sum(abs(xi-x)**2)
            W[i]=np.exp(cst*k)
            u+=1
    W=np.diag(W)
    theta=np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(Y)
    y=theta*x
    Y_pred=[]
    for i in y:
        Y_pred.append(np.sum(i))
    return Y_pred    
       
    





        