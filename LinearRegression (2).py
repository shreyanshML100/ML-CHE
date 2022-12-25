#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np  
import math
import matplotlib.pyplot as plt

# let theta as w
def costFun(w, X_train, Y_train):
    
    # m : Number of training examples
    m = X_train.shape[0]
    cost = 0

    for i in range(m):
        y_hat = w * X_train[i]
        cost = cost + (y_hat - Y_train[i])**2
        J = 1/ (2 * m) * cost

    return J


def diffCostFun(w, X_train, Y_train): 

    m = len(X_train)
    
    # dJ_dw : The gradient of the cost w.r.t. the parameter w
        
    dJ_dw = 0
    
    for i in range(m):  
        y_hat = w * X_train[i] 
        dJ_dw_i = (y_hat - Y_train[i]) * X_train[i] 
        dJ_dw += dJ_dw_i 
    dJ_dw = dJ_dw / m 
        
    return dJ_dw


def FitGD(w,X_train,Y_train,alpha,lambdaa,t,N):
    m=len(X_train)
    w1=np.array(w)
    J=[]
    iter=[]

    if t==1:
        u=1
        while u <= N:
            p=(lambdaa/m)*(np.sum(abs(w1)))
            k=costFun(w, X_train, Y_train) + p
            J.append(k)
            iter.append(u)
            grad = diffCostFun(w, X_train, Y_train)+(lambdaa/m)
            w = w-((alpha)*grad)
            u+=1

    if t==2:
        u=1
        while u <= N:
            p=(lambdaa/(2*m))*(np.sum((w1**2))
            k=costFun(w, X_train, Y_train)+p
            J.append(k)
            iter.append(u)
            grad=diffCostFun(w, X_train, Y_train)+(lambdaa/m)*(w1)
            w=w-((alpha)*grad)
            u+=1

    if t==3:
        u=1
        while u <= N :
            s=(lambdaa/m)*np.sum(abs(w1))
            p=(lambdaa/(2*m))*np.sum((w1)**2)
            k=costFun(w,X_train,Y_train)+p+s
            J.append(k)
            iter.append(u)
            q=(lambdaa/m)*(w1)
            q1=(lambdaa/m)
            grad=diffCostFun(w, X_train, Y_train)+q1+q
            w=w-((alpha)*grad)
            u+=1

    plt.plot(iter,J)
    plt.ylabel("cost Function ( J )")
    plt.xlabel("No. of Iterations")
    plt.show()
    return w
                                      
                                      
def FitNormal(X_train,Y_train):
    
    theta = np.linalg.inv(np.dot(np.transpose(X_train),X_train)),np.transpose(X_train),Y_train
    
    return theta

                                      
                                      
def wm(X_train, tau, x):
    m = X_train.shape[0]
    w = np.mat(np.eye(m))

    for i in range(m):
        xi = X_train[i]
        d = (-2 * tau * tau)
        w[i, i] = np.exp(np.dot((xi - x), (xi - x).T) / d)

    return w


def locallyWeighted(X_train, Y_train, x, tau):
    x_ = np.array([])
    w = wm(X_train, tau, x_)
    theta = np.linalg.pinv(X_train.T * (w * X_train)) * (X_train.T * (w * Y_train))
    y = np.dot(x_, theta)

    return y
                               
                               
# X_train = [[2.5], [4.7], [5.2], [7.3], [9.5], [11.5]]
# Y_train = [5.21, 7.70, 8.30, 11, 14.5, 15]
# w = [1,1]

