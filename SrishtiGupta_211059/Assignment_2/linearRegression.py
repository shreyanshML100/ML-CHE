import numpy as np
import matplotlib.pyplot as plt

def CostFun(theta, X_train, Y_train):
    X=[]
    for f in X_train:
        v=f.copy()
        v.append(1)
        X.append(v)
    sum=0
    Y_train=np.array(Y_train)
    theta=np.array(theta)
    for o in range(len(X)):
        k=np.array(X[o])
        a=theta*k
        b=np.sum(a)-Y_train[o]
        sum+=((b**2))
    return sum/(2*len(X))

def DiffCostFun(theta, X_train, Y_train):
    X=[]
    for f in X_train:
        v=f.copy()
        v.append(1)
        X.append(v)
    diffj=[]
    Y_train=np.array(Y_train)
    theta=np.array(theta)
    for j in range(len(X[0])):
        sum=0
        for o in range(len(X)):
            i=X[o]
            k=np.array(i)
            a=theta*k
            b=(np.sum(a)-Y_train[o])*i[j]
            sum=sum+b
        diffj.append(sum/len(X))
    diffj=np.array(diffj)
    return diffj

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

def FitNormal(X_train,Y_train):
    Y=np.array(Y_train)
    X=[]
    for f in X_train:
        v=f.copy()
        v.append(1)
        X.append(v)
    X=np.array(X)
    theta=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    return theta

def LocallyWeighted(X_train,Y_train,x,tau,no_ofiter):
    x.append(1)
    x=np.array(x)
    X=[]
    for f in X_train:
        v=f.copy()
        v.append(1)
        X.append(v)
    X=np.array(X)
    Y=np.array(Y_train)
    m=len(X)
    n=len(X[0])
    W=[1]*m
    u=0
    while(u<=no_ofiter):
        for i in range (m) :
            xi=np.array(X[i])
            cst=(-1)/(2*(tau**2))
            k=np.sum(abs(xi-x))
            W[i]=np.exp(cst*k)
            u+=1
    W=np.diag(W)
    theta=np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(Y)
    y=theta*x
    return np.sum(y)