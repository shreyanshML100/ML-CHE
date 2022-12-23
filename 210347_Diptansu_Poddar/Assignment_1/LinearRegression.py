import numpy as np
import matplotlib.pyplot as plt


def h(theta,X):
    return np.dot(theta,np.transpose(X))

def costFun(theta, X_train, Y_train):
    sum = 0
    print(Y_train.shape[0])
    for i in range(Y_train.shape[0]):
        sum += 1/(2*(Y_train.shape[0]))*(pow(h(theta,X_train[i])-Y_train[i],2))
    return sum

def diffcostFun(theta, X_train, Y_train):
    sum = np.zeros(len(theta))
    print(Y_train.shape[0])
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
    print(m)
    n = len(theta)
    print(theta)
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
            print(theta)
            if i < iterations+1:
                J_history.append(costFun(theta, X_train, Y_train) + reg_cost)
                p_history.append([theta])

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
            if i < 100000:
                J_history.append(costFun(theta, X_train, Y_train) + reg_cost)
                p_history.append([theta])

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


def locallyWeighted(X_train, Y_train, x, tau):
    x_ = np.array([1,x])
    w = wm(X_train, tau, x_)
    theta = np.linalg.pinv(X_train.T * (w * X_train)) * (X_train.T * (w * Y_train))
    y = np.dot(x_, theta)

    return y

def fitNormal(X_train,Y_train):
    return np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X_train),X_train)),np.transpose(X_train)),Y_train)

X_train = np.array([[2.5, 4.7, 5.2, 7.3, 9.5, 11.5]])
Y_train = np.array([[5.21, 7.70, 8.30, 11, 14.5, 15]])
X_train=np.transpose(X_train)
m = X_train.shape[0]
X_train = np.concatenate([np.ones(m).reshape(m, 1),X_train],axis=1)
Y_train=np.transpose(Y_train)
print(X_train)
#print(locallyWeighted(X_train,Y_train,5,0.5))
theta_in=np.ones(X_train.shape[1])
print(theta_in)
print(fitGD(X_train,Y_train,0.01,0,1,5000,theta_in))
