import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def h(theta,X):
    return np.dot(theta,np.transpose(X))

def costFun(theta, X_train, Y_train):
    sum = 0
    for i in range(Y_train.shape[0]):
        sum += 1/(2*(Y_train.shape[0]))*(pow(h(theta,X_train[i])-Y_train[i],2))
    return sum

def diffcostFun(theta, X_train, Y_train):
    sum = np.zeros(len(theta))
    for j in range(len(theta)):
        for i in range(Y_train.shape[0]):
            sum[j] += 1/(Y_train.shape[0])*(h(theta, X_train[i]) - Y_train[i])*X_train[i][j]
    return sum

    #return 1/(Y_train.shape[0])*sum(h(theta,X_train)-Y_train)*X_train


def fitGD(X_train, Y_train, alpha, lamb_in, ToR, iterations, theta_in):
    J_history = np.zeros(iterations)
    p_history = np.zeros(1)
    theta = theta_in
    m = X_train.shape[0]
    n = len(theta)
    #reg_dj_dtheta = np.zeros(n)

    if ToR == 1:
        for i in range(iterations):
            reg_cost = 0.0
            reg_dj_dtheta = np.zeros(n)
            for j in range(n):
                reg_cost += np.absolute(theta[j])
                reg_dj_dtheta[j] += lamb_in/(2*m)

            reg_cost = (reg_cost * lamb_in)/(2*m)
            dj_dtheta = np.add(diffcostFun(theta, X_train, Y_train), reg_dj_dtheta)
            theta = theta-alpha*dj_dtheta
            J_history[i]  =  (costFun(theta, X_train, Y_train) + reg_cost)

    if ToR == 2:
        for i in range(iterations):
            reg_cost = 0
            reg_dj_dtheta = np.zeros((n,))
            for j in range(n):
                reg_cost += (theta[j] ** 2)
                reg_dj_dtheta[j] += (lamb_in / m) * theta[j]

            reg_cost = (reg_cost * lamb_in) / (2 * m)
            dj_dtheta = np.add(diffcostFun(theta, X_train, Y_train), reg_dj_dtheta)
            theta = theta - alpha * dj_dtheta
            J_history[i] = (costFun(theta, X_train, Y_train) + reg_cost)

    if ToR == 3:
        for i in range(iterations):
            reg_cost = 0
            reg_dj_dtheta = np.zeros((n,))
            for j in range(n):
                reg_cost += 0.5 * (theta[j] ** 2) + 0.5 * np.absolute(theta[j])
                reg_dj_dtheta[j] += 0.5 * (lamb_in / m) * theta[j] + 0.5 * lamb_in / (2 * m)

            reg_cost = (reg_cost * lamb_in) / (2 * m)
            dj_dtheta = np.add(diffcostFun(theta, X_train, Y_train), reg_dj_dtheta)
            theta = theta - alpha * dj_dtheta
            J_history[i] = (costFun(theta, X_train, Y_train) + reg_cost)

    #fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
    #ax1.plot(J_history[:100])
    #ax2.plot(1000 + np.arange(len(J_history[1000:])), J_history[1000:])
    #ax1.set_title("Cost vs. iteration(start)")
    #ax2.set_title("Cost vs. iteration (end)")
    #ax1.set_ylabel('Cost')
    #ax2.set_ylabel('Cost')
    #ax1.set_xlabel('iteration step')
    #ax2.set_xlabel('iteration step')
    plt.plot(J_history)
    plt.show()

    return theta


def wm(X_train, tau, x):
    m = X_train.shape[0]
    w = np.mat(np.eye(m))

    for i in range(m):
        xi = X_train[i]
        d = (-2 * tau * tau)
        w[i, i] = np.exp(np.dot((xi - x), (xi - x).T) / d)

    return w


def locallyWeighted(y,X_train, Y_train, tau):
    x=y
    x_ = np.array([1, x])
    w = wm(X_train, tau, x_)
    theta = np.linalg.pinv(X_train.T * (w * X_train)) * (X_train.T * (w * Y_train))
    y = np.dot(x_, theta)

    return float(y[0][0])

def fitNormal(X_train,Y_train):
    return np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X_train),X_train)),np.transpose(X_train)),Y_train)

"""
X_train = np.array([[2.5, 4.7, 5.2, 7.3, 9.5, 11.5]])
Y_train = np.array([[5.21, 7.70, 8.30, 11, 14.5, 15]])
X_train=np.transpose(X_train)
m = X_train.shape[0]
X_train = np.concatenate([np.ones(m).reshape(m, 1),X_train],axis=1)
Y_train=np.transpose(Y_train)
#print(X_train)
X_test=np.array([1,2,3,4,5])
Y_test=np.zeros(X_test.shape[0])
for i in range(X_test.shape[0]):
    Y_test[i]=locallyWeighted(X_test[i],X_train,Y_train,0.1)
print(Y_test)
"""
import random
def initialize(dim):
    b=random.random()
    theta=np.random.rand(dim)
    return b,theta

def predict_Y(b,theta,X):
    return b + np.dot(X,theta)

def get_cost(Y,Y_hat):
    Y_resd=Y-Y_hat
    return np.sum(np.dot(Y_resd.T,Y_resd))/len(Y-Y_resd)


def update_theta(x,y,y_hat,b_0,theta_o,learning_rate):
    db=(np.sum(y_hat-y)*2)/len(y)
    dw=(np.dot((y_hat-y),x)*2)/len(y)
    b_1=b_0-learning_rate*db
    theta_1=theta_o-learning_rate*dw
    return b_1,theta_1



def run_gradient_descent(X,Y,alpha,num_iterations):
    b,theta=initialize(X.shape[1])
    iter_num=0
    gd_iterations_df=pd.DataFrame(columns=['iteration','cost'])
    result_idx=0
    for each_iter in range(num_iterations):
        Y_hat=predict_Y(b,theta,X)
        this_cost=get_cost(Y,Y_hat)
        prev_b=b
        prev_theta=theta
        b,theta=update_theta(X,Y,Y_hat,prev_b,prev_theta,alpha)
        if(iter_num%10==0):
            gd_iterations_df.loc[result_idx]=[iter_num,this_cost]
        result_idx=result_idx+1
        iter_num +=1
    theta = np.insert(theta, 0, b, axis=0)
    print('Final Estimate of theta : ',theta)
    return gd_iterations_df,theta
#gd_iterations_df,b,theta=run_gradient_descent(X,Y,alpha=0.001,num_iterations=200)
