import numpy as np
import matplotlib.pyplot as plt
def costFun(theta,x,y):
    cost=0
    m=y.shape[0]
    for i in range(m):
        cost+=(y[i]-np.dot(x[i],theta))**2
    
    
    cost=cost/(2*m)
    return cost
def diffCostFun(theta,x,y):
    m=y.shape[0]
    dif=np.matmul(np.transpose(x),((np.matmul(x,theta)-y)))/m
    return dif
def fitGD(x,y,alpha,lambda_,reg,iter):
    theta =np.ones((x.shape[1],1))
    cost=np.zeros(iter)
    # j_=np.zeros(iter)
    for i in range(iter):
        
        cost[i]=costFun(theta,x,y)
        dif=diffCostFun(theta,x,y)
        reg_cost=0
        if reg==1:
            
            for j in range(theta.shape[0]):
                reg_cost+=abs(theta[j])
            reg_cost=reg_cost*lambda_/(theta.shape[0])
            reg_dif=lambda_/(theta.shape[0])*np.ones((theta.shape[0],1))*np.sign(theta)
            dif+=reg_dif
            
        if reg==2:

            for j in range(theta.shape[0]):
                reg_cost+=theta[j]**2
            reg_cost=reg_cost*lambda_/(2*theta.shape[0])
            reg_dif=(lambda_/(theta.shape[0]))*theta
            dif+=reg_dif
            
        if reg==3:
            for j in range(theta.shape[0]):
                reg_cost+=theta[j]**2*0.25+abs(theta[j])*0.5
            reg_cost=reg_cost*lambda_/(theta.shape[0])
            reg_dif=lambda_/(theta.shape[0])*np.ones((theta.shape[0],1))*np.sign(theta)*0.5+lambda_/(theta.shape[0])*theta*0.5
            
            dif+=reg_dif
            
        cost[i]+=reg_cost
        # j_[i]=dif
        theta=theta-alpha*dif

    

    a=np.arange(0,iter)
    
    plt.ylabel('cost')
    plt.xlabel('iter')
    plt.plot(a,cost,marker='x',c='b')
    plt.show()
    return theta
def fitNormal(x,y):
    a=(np.matmul(np.transpose(x),x))
    b=np.matmul(np.linalg.inv(a),np.transpose(x))
    theta=np.matmul(b,y)
    return theta

def locallyWeighted(X_train, Y_train,x ,tau):
    w=np.linalg.pinv(X_train.T@X_train)@X_train.T@Y_train
    res=Y_train-X_train@w
    C=np.diag(res**2)
    theta=np.linalg.pinv(X_train.T@np.linalg.pinv(C)@X_train)@(X_train.T@np.linalg.pinv(C)@Y_train)
    y_predict=np.dot(theta,x)
    return y_predict

        
#     return 
# x=np.array([2.5, 4.7, 5.2, 7.3, 9.5, 11.5])
# y=np.array([5.21, 7.70, 8.30, 11, 14.5, 15])
# x=x.reshape((6,1))
# y=y.reshape((-1,1))
# a=np.ones((6,1))
# x=np.append(a,x,axis=1)
# wei=fitNormal(x,y)
# # print(wei)
# b=np.ones((5,1))
# x_test=np.array([3.5, 5, 6, 8, 10 ])
# x_test=x_test.reshape((5,1))
# x_test=np.append(b,x_test,axis=1)
# # locallyWeighted(x,y,x_test,0.1)
# # print(x_test)
# # print(np.matmul(x_test,wei))
# the=fitGD(x,y,0.01,1,2,10)
# # print("xtest")
# # print(np.matmul(x_test,the))

        
# y_testlo=locallyWeighted(x,y,x_test,10)
# print(y_testlo)

