import numpy as np
import matplotlib.pyplot as plt

def costFun(theta,X_train,Y_train):
    m=len(X_train)
    return 1/(2*m)*np.sum((pow((np.matmul(X_train,theta)-Y_train),2)))
 

def diffcost(theta,X_train,Y_train):
    return (X_train.T.dot(X_train).dot(theta)- X_train.T.dot(Y_train))


def fitGD(X_train,Y_train,alpha,lamda,regu,itr):
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
     
     plt.plot(iteration,cost_history_list,color="green")
     
     return theta


def fitNormal(X_train,Y_train):
    theta=np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(Y_train)
    return theta

global x

def locallyWeighted(X_train,Y_train,x,tau):
    
        x = np.r_[1,x]
        X_train = np.c_[np.ones(len(X_train)), X_train]
        m = np.shape(X_train)[0]
        y= np.zeros(m)
        xw = X_train.T *( np.exp(np.sum((X_train - x) ** 2, axis=1) / (-2 * (tau **2) )))
        theta = np.linalg.pinv(xw @ X_train) @ xw @ Y_train
        y=x@theta
        return y

