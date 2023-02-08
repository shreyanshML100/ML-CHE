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
        # theta = np.matrix([0])
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
    # print(itr)
    # print(j_theta)
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

def fitGrD(X_train,Y_train,alpha,lamda,regu,itr):
     m = np.shape(X_train)[0]  # total number of samples
     n = np.shape(X_train)[1]  # total number of features

     X_train = np.concatenate((np.ones((m, 1)), X_train), axis=1)
     theta = np.random.randn(n + 1, )

     # stores the updates on the cost function (loss function)
     cost_history_list = []
     iteration=[]
     # iterate until the maximum number of iterations
     for current_iteration in range(itr):  # begin the process

        # compute the dot product between our feature 'X' and weight 'W'
        y_estimated = X_train.dot(theta)

        # calculate the difference between the actual and predicted value
        error = y_estimated - Y_train

        # regularization term
        reg_term=0
        gradient = (1 / m)*( X_train.T.dot(error))
        if regu==1:
           reg_term =(lamda/(2 * m)) * np.sum(np.square(theta))
           gradient = (1 / m)*( X_train.T.dot(error) + (lamda* theta))
        if regu==2:
            penalty_term_diff = np.random.randn(n + 1, )
            reg_term = (lamda/(2 * m)) * np.sum(np.absolute(theta))
            for i in range(n+1):
             if theta[i]>0:
              penalty_term_diff[i]=(lamda)/m
             else:
              penalty_term_diff[i]=(-lamda)/m
            gradient = (1 / m)*( X_train.T.dot(error)) +penalty_term_diff
        if regu==3:
             reg_term= (lamda/(2 * m)) *(alpha*np.sum(np.absolute(theta))+(1-alpha)*np.sum(np.square(theta)))
             penalty_term_diff = np.random.randn(n + 1, )
             for i in range(n+1):
              if theta[i]>0:
               penalty_term_diff[i]=(lamda)/m
              else:
               penalty_term_diff[i]=(-lamda)/m
             gradient = (1 / m)*( X_train.T.dot(error) + (lamda*(1-alpha)*theta))+alpha* penalty_term_diff

        # Now we have to update our weights
        theta = theta - alpha * gradient
        # calculate the cost (MSE) + regularization term
        cost = (1 / 2 * m) * np.sum(error ** 2) + reg_term
        #print(f"cost:{cost}   iteration: {current_iteration}")
        cost_history_list.append(cost)
        iteration.append(current_iteration)
     plt.plot(iteration,cost_history_list,color="red")

     return theta
    
    
    



