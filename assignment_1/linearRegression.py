import numpy as np


def costFun(theta,X_train,Y_train):
    m=len(X_train)
    return 1/(2*m)*np.sum((pow((np.matmul(X_train,theta)-Y_train),2)))
 

def diffcost(theta,X_train,Y_train):
    return (X_train.T.dot(X_train).dot(theta)- X_train.T.dot(Y_train))


def fitGD(X_train,Y_train,alpha,lamda,regu,itr):
   
     m = np.shape(X_train)[0]  # total number of samples
     n = np.shape(X_train)[1]  # total number of features
    
     X_train = np.concatenate((np.ones((m, 1)), X_train), axis=1)
     W = np.random.randn(n + 1, )
     
     # stores the updates on the cost function (loss function)
     cost_history_list = []
   
     # iterate until the maximum number of iterations
     for current_iteration in range(itr):  # begin the process
 
        # compute the dot product between our feature 'X' and weight 'W'
        y_estimated = X_train.dot(W)
 
        # calculate the difference between the actual and predicted value
        error = y_estimated - Y_train
 
        # regularization term
        if regu==1:
           ridge_reg_term = (lambda/(2 * m)) * np.sum(np.square(W))
           gradient = (1 / m)*( X.T.dot(error) + (lambda* W))
           # Now we have to update our weights
           W = W - alpha * gradient
        if regu==2:
           ridge_reg_term = (lambda/(2 * m)) * np.sum(np.absolute(W))
        if regu==3:
           ridge_reg_term= (lambda /(2 * m)) *(alpha*np.sum(np.absolute(W))+(1-alpha)*np.sum(np.square(W)))
        # calculate the cost (MSE) + regularization term
        cost = (1 / 2 * m) * np.sum(error ** 2) + ridge_reg_term
        print(f"cost:{cost}   iteration: {current_iteration}")
        cost_history_list.append(cost)
     return W

def fitNormal(X_train,Y_train):
    theta=np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(Y_train)
    return theta


def locallyWeighted(X_train,Y_train,x,tau):
    
        x = np.mat([x, 1])
        m = np.shape(X_train)[0]
        X_train = np.c_[np.ones(len(X_train)), X_train]
        W = np.mat(np.eye(m))
        for i in range(m):
          xi = X_train[i]
          denominator = (-2 * tau * tau)
          W[i, i] = np.exp(np.dot((xi-x), (xi-x).T)/denominator)
        theta = np.linalg.pinv(X_train.T*(W * X_train))*(X_train.T*(W * Y_train))
        y = x@theta
        return y
   """ x = np.r_[1, x]
    X_train = np.c_[np.ones(len(X_train)), X_train]
     
    # fit model: normal equations with kernel
    xw = X_train.T * np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * (tau **2) ))
    theta = np.linalg.pinv(xw @ X) @ xw @ Y
    # "@" is used to
    # predict value
    return x0 @ theta"""

