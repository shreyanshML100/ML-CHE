import numpy as np
import matplotlib.pyplot as plt 
import linearRegression as ok

X_train = np.matrix([2.5, 4.7, 5.2, 7.3, 9.5, 11.5])
X_train = np.transpose(X_train)
Y_train = np.matrix([5.21, 7.70, 8.30, 11, 14.5, 15])
Y_train = np.transpose(Y_train)
theta = np.matrix([0])
alpha=0.01
lam=1
type_=2
NumberofIterations=75
print(ok.costFun(theta, X_train, Y_train))
print(ok.fitNormal(X_train,Y_train))
print(ok.fitGD(X_train, Y_train, alpha, lam,type_, NumberofIterations))