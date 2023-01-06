import numpy as np
import matplotlib.pyplot as plt

def costFun(theta, X_train, Y_train):
    m = X_train.shape[0]
    X_train = X_train + 1
    cost = np.dot(np.transpose(np.dot(X_train,theta)-Y_train),(np.dot(X_train,theta) - Y_train))
    J = (1/(2*m))*cost
    return J ## The Cost Function Value as told in Class.

def diffCostFun(theta, X_train, Y_train):
    m = X_train.shape[0]
    Z = X_train.transpose()
    gradient_of_J = (1/m)*(np.matmul(Z,X_train)*theta - np.matmul(Z,Y_train))
    return gradient_of_J ## The gradient Vector as discussed in class

def fitGD(X_train, Y_train, alpha, lamda, Type_of_Regularization, Iterations):
    j_theta = np.array([])
    itr =np.array([])
    theta = np.matrix([0])
    if Type_of_Regularization==1 :
        theta = np.matrix([0])
        for i in range(Iterations):
            theta =theta + -1*alpha*(diffCostFun(theta,X_train,Y_train)+lamda/(2*len(np.ravel(X_train))))
            itr = np.append(itr,i)
            j_theta = np.append(j_theta,theta)
            
    elif Type_of_Regularization==2 :
        theta = np.matrix([0])
        for i in range(Iterations):
            theta = theta+ -1*alpha*(diffCostFun(theta,X_train,Y_train)+theta*lamda/(len(np.ravel(X_train))))
            itr = np.append(itr,i)
            j_theta = np.append(j_theta,theta)
            
    elif Type_of_Regularization==3 :
        theta = np.matrix([0])
        for i in range(Iterations):
            theta =theta+ -1*alpha*(diffCostFun(theta,X_train,Y_train)+0.5*theta*lamda/(len(np.ravel(X_train)))+0.5*lamda/(2*len(np.ravel(X_train))))
            itr = np.append(itr,i)
            j_theta = np.append(j_theta,theta)
            
    f=plt.figure(1)
    plt.title("Fitting graph")
    plt.plot(itr,j_theta)
    f.show()
    #      This Function performs batch Gradient Descent for our training dataset with the
    #      given parameters as input. It should return the trained parameters and should also plot
    #      the graph of J(theta) vs iteration number for each time it runs using Batch Gradient
    #      Descent. Use vectorized implementation.

    return theta

def fitNormal(X_train, Y_train):
    Z=np.transpose(X_train)
    theta=np.linalg.inv(Z.dot(X_train)).dot(Z.dot(Y_train))
        ##This Function learns the parameter theta using the Normal Equations method.
    return theta


def locallyWeighted(X_train, Y_train, x, tau):
    x = np.r_[1,x]
    X_train = np.c_[np.ones(len(X_train)), X_train]
    m = np.shape(X_train)[0]
    y= np.zeros(m)
    xw = X_train.T *( np.exp(np.sum((X_train - x) ** 2, axis=1) / (-2 * (tau **2) )))
    theta = np.linalg.pinv(xw @ X_train) @ xw @ Y_train
    y=x@theta
        #returns prediction of y for given x through locally weighted Regression
    return y

def error_train(Y_train, Y_pred_train):
    m = Y_pred_train.shape[0]
    error = 0
    for i in range(m):
        error = error + (Y_train[i]-Y_pred_train[i]**2)/(2*m)
    return error

def error_test(Y_train, Y_pred_test):
    m = Y_pred_test.shape[0]
    error = 0
    for i in range(m):
        error = error + (Y_test[i]-Y_pred_test[i]**2)/(2*m)
    return error

