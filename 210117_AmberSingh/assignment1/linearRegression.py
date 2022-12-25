import numpy as np
def costFun(theta,X_train,Y_train):

    X_train=np.array(X_train)
    Y_train=np.array(Y_train)

    m=X_train.shape[0]
    cost_sum=0
    for i in range(m):
        
        h=theta[i]*X_train[i]
        cost=(h - Y_train[i]) ** 2  
        cost_sum=cost_sum+cost
        J=(1/(2*m))*cost_sum
    return J

def diffCostFun(theta,X_train,Y_train): # we need to return XT X theeta -XT y
    X_train=np.array(X_train)
    Y_train=np.array(Y_train)

    X_train=np.array(X_train)


    Z=X_train.transpose()
    a=(np.dot(Z,X_train)*theta)
    b=np.dot(Z,Y_train)
    gradient=np.subtract(a,b)
    
    return gradient
def fitGD(X_train,Y_train,alpha,theta,type,lamda,num_iters):
    
    if type==1:
        m = len(Y_train)
        cost_history = np.zeros(num_iters)
        gradient=diffCostFun(theta, X_train, Y_train)
        
        cost=costFun(theta,X_train,Y_train)
        
        for i in range(num_iters):
            cost += (lamda/m) * np.sum(np.abs(theta[1:]))
            gradient[1:] += (lamda/m) * np.sign(theta[1:])
            grad = gradient
            theta -= alpha * grad
        cost_history[i] = cost

        
    return theta,cost_history
    
    if type==2:
        m = len(Y_train)
        cost_history = np.zeros(num_iters)
        gradient=diffCostFun(theta, X_train, Y_train)
        
        cost=costFun(theta,X_train,Y_train)
        
        for i in range(num_iters):
            gradient[1:] += (lamda/m)  * theta[1:]
            cost=cost+ (lamda/m) * np.sum(theta[1:]**2)
            grad = gradient
            theta -= alpha * grad
            cost_history[i] = cost

        
    return theta,cost_history
    if type==3:

        m = len(Y_train)
        cost_history = np.zeros(num_iters)
        gradient=diffCostFun(theta, X_train, Y_train)
        cost=costFun(theta,X_train,Y_train)
        for i in range(num_iters):
            gradient[1:] +=  (lamda/m) * ((1 - alpha) * theta[1:] + alpha * np.sign(theta[1:]))
            cost=cost + (lamda/m) * ((1 - alpha) * np.sum(theta[1:]**2) + alpha * np.sum(np.abs(theta[1:])))
            grad = gradient
            theta -= alpha * grad
            cost_history[i] = cost
        
    return theta,cost_history

def fitNormal(X_train,Y_train):
    X_train=np.array(X_train)
    Y_train=np.array(Y_train)
    np.insert(X_train,0,1)
    np.insert(Y_train,0,1)
    Z=X_train.transpose()
    a=np.dot(np.dot(Z,X_train))
    ainv = np.linalg.inv(a) 
    b=np.dot(Z,Y_train)
    theeta=np.dot(a,b)
    return theeta

def locallyWeighted(X_train, Y_train, x_pred, tau, num_iters=1):
    X_train=np.array(X_train)
    Y_train=np.array(Y_train)

    y_pred = np.mean(Y_train)
    for i in range(num_iters):
        distances = np.sqrt(np.sum((X_train - x_pred)**2, axis=1))
        weights = np.exp(-distances**2 / (2 * tau**2))
        y_pred = np.sum(weights * Y_train) / np.sum(weights)
        return y_pred   
       

X_train = [[2.5], [4.7], [5.2], [7.3], [9.5], [11.5]]
Y_train = [5.21, 7.70, 8.30, 11, 14.5, 15]
X_train=np.array(X_train)
Y_train=np.array(Y_train)
m=X_train.shape[0]
theta=np.ones(m)
type=1
num_iters=75
alpha=0.01
lamda=2
a=fitGD(X_train,Y_train,alpha,theta,1,lamda,num_iters)
