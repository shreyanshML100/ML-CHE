import numpy as np
import matplotlib.pyplot as plt
    
def fit_Normal(X_train,Y_train):
    X_train_transpose=X_train.transpose()
    invertible_matrix=np.matmul(X_train_transpose,X_train)
    inverse=np.linalg.inv(invertible_matrix)
    next=np.matmul(inverse,X_train_transpose)
    return np.matmul(next,Y_train)

def cost_Fun(theta,X_train,Y_train):
    a=np.matmul(X_train,theta)-Y_train
    b=a.transpose()
    return np.matmul(b,a)

def diff_costfun(theta,X_train,Y_train):
    rows,cols=X_train.shape
    finale=(-1*(np.matmul(X_train,theta))+Y_train)
    sum=np.zeros((1,cols))
    for i in range(rows):
        sum = sum + finale[i]*X_train[i]
    return sum

def fit_GD(X_train,Y_train,alpha,Lambda,regularize,iteration_num):
    
    if regularize==1:
        rows,cols=X_train.shape
        new_lambda=(Lambda*0.5/rows)*(-1)*(alpha)
        theta=np.zeros((cols,1))
        for i in range(iteration_num):
            fun=(alpha/rows)*diff_costfun(theta,X_train,Y_train)
            new_theta=theta+fun.transpose()
            for j in range(cols):
                if theta[j]<0:
                    theta[j] = new_theta[j]-new_lambda
                elif theta[j]>=0:
                    theta[j] = new_theta[j]+new_lambda

    elif regularize==2:
        rows,cols=X_train.shape
        new_lambda=(Lambda*0.5/rows)*(-1)*(alpha)
        theta=np.zeros((cols,1))
        for i in range(iteration_num):
            fun=(alpha/rows)*diff_costfun(theta,X_train,Y_train)
            new_theta=theta+fun.transpose()
            theta=new_theta+2*new_lambda*theta
    
    return theta

def fit_GD_2(X_train,Y_train,alpha,Lambda,ratio,iteration_num):
    rows,cols=X_train.shape
    new_lambda=(Lambda*0.5/rows)*(-1)*(alpha)
    theta=np.zeros((cols,1))
    for i in range(iteration_num):
        fun=(alpha/rows)*diff_costfun(theta,X_train,Y_train)
        new_theta=theta+fun.transpose()
        for j in range(cols):
            if theta[j]<0:
                theta[j] = new_theta[j]-(ratio*new_lambda)+(((1-ratio)*new_lambda*2)*theta[j])
            elif theta[j]>=0:
                theta[j] = new_theta[j]+(ratio*new_lambda)+(((1-ratio)*new_lambda*2)*theta[j])
    return theta

def locally_weighted(X_train,Y_train,X_test,tau):
    rows,cols=X_train.shape
    r,c=X_test.shape
    weights = np.mat(np.eye(rows))
    weights=np.identity(rows)
    for j in range(r):
        for i in range(rows):
            z=X_train[i]-X_test[j]
            print(z)
            weights[i][i]=np.exp((-1/(2*tau*tau))*np.matmul(z.transpose(),z))
    X_train_transpose=X_train.transpose()
    m1=np.matmul(X_train_transpose,weights)
    m2=np.matmul(m1,X_train)
    inverse=np.linalg.inv(m2)
    next=np.matmul(inverse,X_train_transpose)
    m3=np.matmul(next,weights)
    theta_final=np.matmul(m3,Y_train)
    return np.matmul(theta_final.transpose(),X_test.transpose())

def error_func(actual,pred):
    rows,cols=actual.shape
    err=np.zeros((rows,1))
    for i in range(rows):
        err[i]=(actual[i]-pred[i])*(actual[i]-pred[i])
    return err/(rows*2)

