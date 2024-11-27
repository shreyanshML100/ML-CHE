#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import seaborn as sns
import datetime


# In[ ]:


def costFun(Theta,x_train,y_train):
    s=np.sum((np.matmul(Theta,x_train)-y_train)**2);
    return (1/(2*y_train.shape[1]))*s;

def diffCostFun(Theta,x_train,y_train):
    b=0;
    b=np.dot((np.matmul(Theta,x_train)-y_train),np.transpose(x_train));
    b=b/(y_train.size);
    return b;
  
def fitGD(x_train,y_train,alpha,lamda,Type,iterations,pt):
  n=x_train.shape[0]
  Theta=np.ones(n,dtype=float);
  m=y_train.shape[1];
  # cost=np.zeros(iterations);
  
  for itr in range(iterations):
      if Type==1:
          r1=0.;
          r2=np.zeros(2);
          r1=(np.sum(np.absolute(Theta)))*lamda/(2*m);
          r2=(lamda/(2*m));
          k=diffCostFun(Theta,x_train,y_train)+r2;
          Theta=Theta-alpha*k;

      if Type==2:
          r1=0;
          r2=np.zeros(n);
          r1=np.sum(np.square(Theta))*lamda/m;
          r2=(lamda/m)*(Theta);
          k=diffCostFun(Theta,x_train,y_train)+r2;
          Theta=Theta-alpha*k;

      if Type==3:
          r1=0;
          r1=0.5*(np.square(Theta)+np.absolute(Theta));
          r2=np.zeros(2);
          r2=(lamda/m)*(0.5*(Theta+0.5*np.ones(Theta.size)));
          k=diffCostFun(Theta,x_train,y_train)+r2;
          Theta=Theta-alpha*k;
        # cost[itr]=(costFun(Theta,x_train,y_train)+r1);

  if pt==1:
      plt.plot(cost)
      plt.show()
      plt.scatter(x_train[1],y_train);
      x = np.linspace(0,12,100)
      y=Theta[0,1]*x+Theta[0,0];
      plt.plot(x,y,"r-");
      plt.show()
  return Theta;

def fitNormal(x_train,y_train):
    yn= np.transpose(y_train);
    xn=np.transpose(x_train);
    return np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(xn),xn)),np.transpose(xn)),yn)



df = pd.read_excel('AirQualityUCI.xlsx')
d1=df

df['HOUR']=df['Time'].apply(lambda x: int(x.strftime('%H')))
df.HOUR.head()

df['Date']=pd.to_datetime(df.Date, format='%m/%d/%Y')
df.set_index('Date',inplace=True)
df['Month']=df.index.month
df.reset_index(inplace=True)
df.head()
# for i in range(0,9357):
#   df.loc[i,"Time"]=df.loc[i,"Time"].strftime("%H")


# df

df.info()
df.shape

df.describe()

df.duplicated().sum()

df.isnull().sum()

fig = plt.subplots(figsize=(10,10)) 
sns.heatmap(df.corr(),annot=True)
plt.title('Heatmap of co-relation between variables',fontsize=16)
plt.show()

#Plots of various features against output variable(RH)

col=df.columns.tolist()[2:]
for i in df.columns.tolist()[2:]:
    sns.lmplot(x=i,y='RH',data=df,markers='.')

ar=df.to_numpy()

scaler = sklearn.preprocessing.StandardScaler()
Y=ar[:,13]
x=ar[:,2:17]
scaler.fit(ar[:,2:17])
x=scaler.transform(ar[:,2:17])
x1=x[:,0:11]
x2=x[:,13:15]
cp=x
X=np.append(x1,x2,axis=1)

print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y ,test_size=0.10, random_state=42)
Y_train=np.array([Y_train])
Y_test=np.array([Y_test])
X_train=np.transpose(X_train)
X_train=np.append([np.ones(Y_train.size)],X_train,axis=0)
X_test=np.transpose(X_test)
X_test=np.append([np.ones(Y_test.size)],X_test,axis=0)
X_test=np.transpose(X_test)

theta=fitGD(X_train,Y_train,0.01,2,2,100,0)
Y_pred1=np.dot(theta,X_train)
Y_pred2=np.dot(theta,np.transpose(X_test))

Theta=fitNormal(X_train,Y_train)
Y_Pred1=np.dot(np.transpose(X_train),Theta)
Y_Pred2=np.dot(X_test,Theta)

print(theta)
print(Theta)

J_train=np.sum((Y_train-Y_pred1)**2/(2*Y_train.shape[0]))
print(J_train)
J_test=np.sum((Y_test-Y_pred2)**2/(2*Y_test.shape[0]))
print(J_test)
J_trainN=np.sum((Y_train-Y_Pred1)**2/(2*Y_train.shape[0]))
print(J_trainN)
J_testN=np.sum((Y_test-Y_Pred2)**2/(2*Y_test.shape[0]))
print(J_testN)

print(Y_test)
print(Y_pred2)

