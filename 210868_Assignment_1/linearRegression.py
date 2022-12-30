import numpy as np
import matplotlib.pyplot as plt

def costFun(theta,X_train,Y_train,reg,lambda_):
    b=theta[0]
    w=theta[1:]
    m,n=X_train.shape
    total_cost=0
    for i in range(m):
        f_w_b=np.dot(X_train[i],w)+b
        cost=(f_w_b-Y_train[i])**2
        total_cost=total_cost+cost
    if reg==0:
        return total_cost/(2*m)
    elif reg==1:
        sum=0
        for j in range(n):
            sum=sum+abs(w[j])
        total_cost=total_cost+lambda_*(sum)/(2*m)
    elif reg==2:
        sum=0
        for j in range(n):
            sum=sum+w[j]**2
        total_cost=total_cost+lambda_*(sum)/(2*m)
    elif reg==3:
        sum=0
        for j in range(n):
            sum=sum+lambda_*abs(w[j])/(2*m)+lambda_*(w[j]**2)/(2*m)
        total_cost=total_cost+sum
    

def diffCostFun(theta,X_train,Y_train,reg,lambda_):
    b=theta[0]
    w=theta[1:]
    m,n=X_train.shape
    dj_dw=np.zeros(n)
    dj_db=0
    for i in range(m):
        f_w_b=np.dot(X_train[i],w)+b
        err=f_w_b-Y_train[i]
        for j in range(n):
            dj_dw[j]=dj_dw[j]+err*X_train[i][j]
        dj_db=dj_db+err
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    grad=np.empty_like(theta)
    if reg==0:
            grad[0]=dj_db
            grad[1:]=dj_dw
            return grad
    elif reg==1:
        grad[0]=dj_db
        dj_dw=dj_dw+lambda_/(2*m)
        grad[1]=dj_dw
        return grad

    elif reg==2:
        grad[0]=dj_db
        dj_dw=dj_dw+lambda_*w/(m)
        grad[1]=dj_dw
        return grad
    elif reg==3:
        grad[0]=dj_db
        dj_dw=dj_dw+lambda_*w/(m)+lambda_/(2*m)
        grad[1]=dj_dw
    return grad

def fitGD(X_train,Y_train,alpha,lambda_,Regu,iter):
    j=[] 
    m,n=np.shape(X_train)
    theta=np.zeros(n+1)
    for k in range(iter):
        j.append(costFun(theta,X_train,Y_train,Regu,lambda_))
        grad=diffCostFun(theta,X_train,Y_train,Regu,lambda_)
        theta=theta-alpha*grad
        # if k%1000==0:
        #     print("Number of Iterations:{}".format(k))
        #     print("Cost Function:{}".format(j[-1]))
    plt.plot(j)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost')
    return theta

def fitNormal(X_train,Y_train):
    X_train_t=np.transpose(X_train)
    theta=np.linalg.inv(X_train_t.dot(X_train)).dot(X_train_t.dot(Y_train))
    return theta

def locallyWeighted(X_train,Y_train,x,tau,iter):
    y=np.empty_like(x)
    m,n=np.shape(X_train)
    X_train_t=np.transpose(X_train)
    tr=np.shape(x)[0]
    for i in range(tr):
        weight=np.mat(np.eye(m))
        for j in range(m):
            diff=x[i]-X_train[j]
            weight[j,j]=np.exp(-(diff**2)/(2*tau**2))
        W=(np.linalg.pinv(X_train_t*weight*X_train))*X_train_t*weight*Y_train
        y[i]=x[i]*W
    return y





