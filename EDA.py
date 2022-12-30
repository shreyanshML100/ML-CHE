#!/usr/bin/env python
# coding: utf-8

# In[70]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[2]:


df = pd.read_excel('AirQualityUCI.xlsx')


# In[22]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.describe()


# In[8]:


df.nunique()


# In[9]:


df.columns


# In[10]:


sns.pairplot(data=df)


# In[11]:


nullvalues = df.isnull().sum().sort_values(ascending=False)


# In[12]:


nullvalues


# In[13]:


corelation = df.corr()


# In[14]:


sns.heatmap(corelation, xticklabels = corelation.columns, yticklabels = corelation.columns, annot=True)


# ## Data Preprocessing

# In[55]:


for i in range(df.shape[0]):
    df.replace(df['Time'][i], df['Time'][i].hour, inplace= True)


# In[57]:


month = []
year = []
for i in range(df.shape[0]):
    month.append(df['Date'][i].month)
    year.append(df['Date'][i].year)


# In[58]:


df.insert(0, "Month", month)
df.insert(1, "Year", year)


# In[59]:


df.drop(["Date"], axis = 1, inplace = True)


# In[60]:


dataplot2 = pd.DataFrame(df.corr())


# In[61]:


dataplot2['RH'] #since year has very less correlation remove year


# In[62]:


df.drop(["Year"], axis = 1, inplace = True)


# In[63]:


columns = np.array(df.columns)
new_col = ['Month', 'Time', 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
       'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)',
       'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'AH','RH']
df = df.reindex(columns = new_col)


# In[65]:


df.duplicated().sum() #no duplicate values


# In[67]:


array = df.to_numpy()


# In[68]:


df


# In[71]:


scaler = StandardScaler()


# In[72]:


X = array[:, :14]
Y = array[:, 14]


# In[73]:


len(Y) == len(X)


# In[74]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, train_size=  0.9)


# In[75]:


scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# ## Model

# In[76]:


import sys


# In[98]:


sys.path.append("LinearRegression.py")


# In[99]:


import LinearRegression as lr
from LinearRegression import *


# In[100]:


def error(y, yPred):
    ev = y - yPred
    error = 0
    for ele in ev:
        error = error + ele*ele
    return error/(2*(ev.shape[0]))


# In[101]:


theta = fitGD(X_train,Y_train, 1e-5, 0.01, 0, 1000)
y_Pred = np.matmul(np.hstack((np.ones((X_test.shape[0], 1)), X_test)),theta).reshape(Y_test.shape[0])
error(Y_test, y_Pred)


# In[102]:


theta3 = fitGD(X_train,Y_train, 1e-5, 0.01, 0, 1000)
y_Pred3 = np.matmul(np.hstack((np.ones((X_train.shape[0], 1)), X_train)),theta3).reshape(Y_train.shape[0])
error(Y_train, y_Pred3)


# In[103]:


y_Pred


# In[104]:


y_Pred3


# ## Normal Equation

# In[105]:


theta2 = fitNormal(X_train, Y_train)
y_Pred2 = np.matmul(np.hstack((np.ones((X_test.shape[0], 1)), X_test)),theta2).reshape(Y_test.shape[0])
error(Y_test, y_Pred2)


# In[106]:


y_Pred2 

