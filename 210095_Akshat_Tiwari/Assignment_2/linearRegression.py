import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as m


def costFun(theta, X_train, Y_train):
    m = len(X_train)
    mat = np.dot(X_train, theta) 
    mat = np.subtract(mat,Y_train)
    J = np.dot(np.transpose(mat), mat)
    J /= (2*m)
    return J

def diffCostFun(theta, X_train, Y_train):
     def sum(a,b):
        return (a+b)
     mat = np.dot(X_train, theta) 
     mat = np.subtract(mat,Y_train)
     grad = np.matmul(np.transpose(X_train),mat) 
     grad/= len(np.ravel(X_train))
     return grad 


def fitNormal(X_train, Y_train):
    X_train_t = np.transpose(X_train)
    a = np.matmul(X_train_t, X_train)
    a = np.linalg.inv(a)
    b = np.matmul(a, X_train_t)
    theta = np.matmul(b, Y_train)
    return theta
def fitGD(X_train, Y_train, alpha, lam, TypeofRegularization, NumberofIterations) :
    j_theta = np.array([])
    itr =np.array([])
    if TypeofRegularization==1 :
        theta = np.matrix([0])
        for i in range(NumberofIterations):
            theta =theta + -1*alpha*(diffCostFun(theta,X_train,Y_train)+lam/(2*len(np.ravel(X_train))))
            itr = np.append(itr,i)
            j_theta = np.append(j_theta,theta)
    elif TypeofRegularization==2 :
        theta = np.matrix([0])
        for i in range(NumberofIterations):
            theta = theta+ -1*alpha*(diffCostFun(theta,X_train,Y_train)+theta*lam/(len(np.ravel(X_train))))
            itr = np.append(itr,i)
            j_theta = np.append(j_theta,theta)
    elif TypeofRegularization==3 :
        theta = np.matrix([0])
        for i in range(NumberofIterations):
            theta =theta+ -1*alpha*(diffCostFun(theta,X_train,Y_train)+0.5*theta*lam/(len(np.ravel(X_train)))+0.5*lam/(2*len(np.ravel(X_train))))
            itr = np.append(itr,i)
            j_theta = np.append(j_theta,theta)
    X_train = np.matrix([2.5, 4.7, 5.2, 7.3, 9.5, 11.5])
    X_train = np.transpose(X_train)
    Y_train = np.matrix([5.21, 7.70, 8.30, 11, 14.5, 15])
    Y_train = np.transpose(Y_train)
    print(itr)
    print(j_theta)
    # plt.title("Fitting graph")
    # plt.scatter(np.ravel(X_train),np.ravel(Y_train),color="red")
    f=plt.figure(1)
    plt.title("Fitting graph")
    plt.plot(itr,j_theta,color="red")
    f.show()
    return (theta)

def locallyWeighted(X_train, Y_train, x, tau, NumberofIterations):
    x = np.r_[1,x]
    X_train = np.c_[np.ones(len(X_train)), X_train]
    m = np.shape(X_train)[0]
    y= np.zeros(m)
    xw = X_train.T *( np.exp(np.sum((X_train - x) ** 2, axis=1) / (-2 * (tau **2) )))
    theta = np.linalg.pinv(xw @ X_train) @ xw @ Y_train
    y=x@theta
    return y
    
    
    



