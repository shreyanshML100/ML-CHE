import numpy as np
from numpy.linalg import inv
import math
import matplotlib.pyplot as plt


def costFun(X_train,Y_train,theta):
m=x.shape[0]
J=0
x0=np.ones(m).reshape(m,1)
x=np.concatenate((x0,x), axis=1)
loss=np.dot(x,theta)-y
J=np.sum(loss**2)/(2*m)
return J   


def diffCostFun(theta, X_train, Y_train):
    x = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    Theta = theta.reshape((theta.shape[0], 1))
    xTrans = x.T
    y = Y_train.reshape((Y_train.shape[0], 1))
    return np.matmul(xTrans, np.matmul(x, Theta)) - np.matmul(xTrans, y)


def fitNormal(X_train, Y_train):
    y = Y_train.reshape((Y_train.shape[0], 1))
    x = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    xTrans = x.T
    return np.dot(np.matmul(inv(np.matmul(xTrans, x)), xTrans), y)


def fitGD(X_train, Y_train, alpha, lam, reg, iterations):
    y = Y_train.reshape((Y_train.shape[0], 1))
    x = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

    m = x.shape[0]  # examples
    n = x.shape[1]  # features
    storeJ = []

    theta = np.zeros((n, 1))
    if reg == 0: #no regularization
        for i in range(iterations):
            theta = theta - alpha * (diffCostFun(theta, X_train, Y_train))
            storeJ.append(costFun(theta, X_train, Y_train))


    elif reg == 1: #ridge regularization
        for i in range(iterations):
            theta = theta - alpha * (diffCostFun(theta, X_train, Y_train) + lam*theta)
            storeJ.append(costFun(theta, X_train, Y_train))

    elif reg == 2: #lasso regularization
        for i in range(iterations):
            mod = theta/abs(theta)
            for i in range(mod.shape[0]):
                if np.isnan(mod[i]):
                    mod[i] = 0
            theta = theta - alpha * (diffCostFun(theta, X_train, Y_train) + lam*mod)
            storeJ.append(costFun(theta, X_train, Y_train))
    plt.plot(range(iterations), storeJ)
    plt.xlabel('iterations')
    plt.ylabel('Cost function')
    return theta

def locallyWeighed(X_train, Y_train, x, alpha, tau, iterations):
    y = Y_train.reshape((Y_train.shape[0], 1))
    X = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    xC = x
    xC = np.hstack((np.ones((x.shape[0], 1)), xC))

    m = X.shape[0]  # examples
    n = X.shape[1]  # features
    mTest = xC.shape[0]
    yReturn = []

    for k in range(mTest):
        theta = np.zeros((n, 1))
        for j in range(iterations):
            sum = np.zeros((1, n))
            for i in range(m):
                W = math.exp(np.dot(X[i] - xC[k], X[i] - xC[k]) / (-2 * tau * tau))
                sum = sum + (W * (y[i] - np.dot(X[i], theta))) * X[i]
            sum = sum.reshape((n, 1))
            theta = theta + alpha * sum
        yReturn.append(np.dot(xC[k], theta))
    return np.array(yReturn)
