import numpy as np
import matplotlib.pyplot as plt
import linearRegression as lr

train_X=np.array([[1,2.5],[1,4.7],[1,5.2],[1,7.3],[1,9.5],[1,11.5]])
train_Y=np.array([[5.21],[7.70],[8.30],[11],[14.5],[15]])
X_test=np.array([[1,3.5],[1,5],[1,6],[1,8],[1,10]])
tau=np.array([[1,10,100]])
lambda_input=np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]])
tau_val=np.array([[0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000]])
Y_test=np.array([[6.25,8.2,9.5,12.1,14.7]])

theta_num=lr.fit_Normal(train_X,train_Y)
Y_normalfit=np.matmul(theta_num.transpose(),X_test.transpose())
plt.plot(X_test[:,1],Y_normalfit.transpose(),marker='o',color='red')
plt.scatter(train_X[:,1],train_Y,color='green')
plt.show()

for i in range(15):
    theta_num_gd1=lr.fit_GD(train_X,train_Y,0.01,lambda_input[0][i],1,100)
    Y_gdfit_1=np.matmul(theta_num_gd1.transpose(),X_test.transpose())
    Y_train_err=np.matmul(theta_num_gd1.transpose(),train_X.transpose())
    plt.plot(X_test[:,1],Y_gdfit_1.transpose(),marker='o',color='black')
    plt.scatter(train_X[:,1],train_Y,color='green')
    plt.show()
    J_train=lr.error_func(train_Y,Y_train_err.transpose())
    J_test=lr.error_func(Y_test.transpose(),Y_gdfit_1.transpose())
    plt.plot(train_X[:,1],J_train,marker='o',color='red')
    plt.plot(X_test[:,1],J_test,marker='X',color='blue')
    plt.show()

for i in range(15):
    theta_num_gd2=lr.fit_GD(train_X,train_Y,0.01,lambda_input[0][i],2,100)
    Y_gdfit_2=np.matmul(theta_num_gd2.transpose(),X_test.transpose())
    Y_train_err_2=np.matmul(theta_num_gd2.transpose(),train_X.transpose())
    plt.plot(X_test[:,1],Y_gdfit_2.transpose(),marker='o',color='black')
    plt.scatter(train_X[:,1],train_Y,color='green')
    plt.show()
    J_train=lr.error_func(train_Y,Y_train_err_2.transpose())
    J_test=lr.error_func(Y_test.transpose(),Y_gdfit_2.transpose())
    plt.plot(train_X[:,1],J_train,marker='o',color='red')
    plt.plot(X_test[:,1],J_test,marker='X',color='blue')
    plt.show()

for k in range(3):
    b=tau[0][k]
    Y_lw=lr.locally_weighted(train_X,train_Y,X_test,b)
    Y_train_x=lr.locally_weighted(train_X,train_Y,train_X,b)
    plt.plot(X_test[:,1],Y_lw.transpose(),marker='X',color='black')
    plt.scatter(train_X[:,1],train_Y,color='green')
    plt.show()
    J_test=lr.error_func(Y_test.transpose(),Y_lw.transpose())
    J_train=lr.error_func(train_Y,Y_train_x.transpose())
    plt.plot(train_X[:,1],J_train,marker='o',color='red')
    plt.plot(X_test[:,1],J_test,marker='X',color='blue')
    plt.show()
