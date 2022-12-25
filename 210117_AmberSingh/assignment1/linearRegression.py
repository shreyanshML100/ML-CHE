import numpy as np



def costFun(theta, X_train, Y_train):
    m = len(X_train)
    a = np.dot(X_train, theta) 
    b = np.subtract(a,Y_train)
    J = (1/2*m) * np.dot(X_train.T,X_train)
    
    return J

def diffCostFun(theta, X_train, Y_train):

    m=Y_train.shape[0]
    dif=np.matmul(np.transpose(X_train),((np.matmul(X_train,theta)-Y_train)))/m
    return dif



def fitGD(X_train, Y_train, alpha, lamda,type_, iter) :
    j_theta = np.array([])
    itr =np.array([])
    if type_==1 : #L1
        theta = np.matrix([0])
        for i in range(iter):
            theta =theta + -1*alpha*(diffCostFun(theta,X_train,Y_train)+lamda/(2*len(np.ravel(X_train))))
            itr = np.append(itr,i)
            j_theta = np.append(j_theta,theta)
    elif type_==2 :#L2
        theta = np.matrix([0])
        for i in range(iter):
            theta = theta+ -1*alpha*(diffCostFun(theta,X_train,Y_train)+theta*lamda/(len(np.ravel(X_train))))
            itr = np.append(itr,i)
            j_theta = np.append(j_theta,theta)
    elif type_==3 : #ELASTIC
        theta = np.matrix([0])
        for i in range(iter):
            theta =theta+ -1*alpha*(diffCostFun(theta,X_train,Y_train)+0.5*theta*lamda/(len(np.ravel(X_train)))+0.5*lamda/(2*len(np.ravel(X_train))))
            itr = np.append(itr,i)
            j_theta = np.append(j_theta,theta)
   
    print(j_theta)
   
    
    return (theta)

def fitNormal(X_train,Y_train):
    X_train=np.array(X_train)
    Y_train=np.array(Y_train)
    np.insert(X_train,0,1)
    np.insert(Y_train,0,1)
    Z=X_train.T
    a=(np.dot(Z,X_train))
    ainv = np.linalg.inv(a) 
    b=Z @ Y_train
    theeta=np.dot(a,b)
    return theeta

def locallyWeighted(X_train, Y_train, x_pred, tau, num_iters):
    X_train=np.array(X_train)
    Y_train=np.array(Y_train)

    y_pred = np.mean(Y_train)
    for i in range(num_iters):
        distances = np.sqrt(np.sum((X_train - x_pred)**2, axis=1))
        weights = np.exp(-distances**2 / (2 * tau**2))
        y_pred = np.sum(weights * Y_train) / np.sum(weights)
        return y_pred   