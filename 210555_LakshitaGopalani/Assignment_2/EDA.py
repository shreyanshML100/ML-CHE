import numpy as np
from numpy import NaN
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import seaborn as sns
import linearRegression as lr

df = pd.read_excel("/content/AirQualityUCI.xlsx")

print(df.info())

print(df.describe())

#convert Date and Time values:
df['Date'] = df['Date'].dt.month
df['Time'] = pd.to_datetime(df['Time'], format='%X').dt.hour

df = df.replace(-200, NaN) #replace -200 with NaN values
print(df.isna().sum())
print(df.duplicated().sum())

df =  df.drop(['NMHC(GT)'], axis=1) #drop column NMHC(GT) since it has too many missing values
df = df.dropna() #drop rows with missing value
df = df.drop_duplicates() 
df = df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,13,12]] #make RH the last column

#plot correlations using a heatmap:
plt.figure(figsize=(14,8))
sns.heatmap(df.corr(), annot = True)

#plot RH vs other variables in different plots:
for i in df.columns:
  sns.lmplot(data=df, x=i, y='RH')

arr = df.to_numpy()
X = arr[:, 0:13]
Y = arr[:, 13]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, shuffle=True, random_state=42)

#standardise data:
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

X_train = np.insert(X_train, 0, 1, axis=1)
X_test = np.insert(X_test, 0, 1, axis=1) #adding bias term

Y_train = Y_train.reshape((Y_train.shape[0], 1))
Y_test = Y_test.reshape((Y_test.shape[0], 1)) #convert row vector to column vector

#Linear Regression using Gradient Descent:
theta = lr.fitGD(X_train, Y_train, 0.01, 0.01, 2, 1000)
Y_pred_train_GD = X_train @ theta
Y_pred_test_GD = X_test @ theta
J_train_GD = lr.costFun(theta, X_train, Y_train)
J_test_GD = lr.costFun(theta, X_test, Y_test)

print("Using Gradient Descent:\n\nError on training data = ", J_train_GD, "\nError on test data = ", J_test_GD, "\n\n")

#Linear Regression using Normal Equations:
theta_n = lr.fitNormal(X_train, Y_train)
Y_pred_train_normal = X_train @ theta_n
Y_pred_test_normal = X_test @ theta_n
J_train_n = lr.costFun(theta_n, X_train, Y_train)
J_test_n = lr.costFun(theta_n, X_test, Y_test)

print("Using Normal Equations:\n\nError on training data = ", J_train_n, "\nError on test data = ", J_test_n, "\n\n")
