```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import keras
import keras_preprocessing
from sklearn.preprocessing import StandardScaler
```


```python
sheet_of_data = pd.read_excel("C:\Users\AMAN KUMAR\AirQualityUCI\AirQualityUCI.xlsx")
```


```python
sheet_of_data.head()
```


```python
sheet_of_data.info()
```


```python
sheet_of_data['Date']=pd.to_datetime(sheet_of_data['Date'],format=' %y-%m-%d')
sheet_of_data=sheet_of_data.drop('Time', axis=1)
sheet_of_data=sheet_of_data.drop('Date',axis=1)
sheet_of_data.head()
```


```python
col_=sheet_of_data.columns.tolist()[1:]
X=sheet_of_data[col_].drop('RH',axis=1)

y=sheet_of_data['RH'] 
ss_x = StandardScaler()
X=ss_x.fit_transform(X) 
```


```python
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.20, random_state=0,shuffle=False)
```


```python
X_train
Xt=X_train
yt=y_train
```


```python
model = tf.keras.Sequential([
    tf.keras.Input(shape=(11,)),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=1, activation='linear')
     ])
```


```python
model.compile(optimizer='adam',
              loss='tf.keras.losses.mean_squared_error')
```


```python
model.compile(optimizer='adam',
             loss=tf.keras.losses.mean_squared_error)
```


```python
model.fit(X_train,y_train, epochs=50)
```


```python
y_predicted = model.predict(X_test)
```


```python
model2 = tf.keras.Sequential([
    tf.keras.Input(shape=(11,)),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=8, activation='relu'),
    keras.layers.Dense(units=4, activation='relu'),
    keras.layers.Dense(units=2, activation='relu'),
    keras.layers.Dense(units=1, activation='linear')
     ])
```


```python
model2.compile(optimizer='adam',
             loss=tf.keras.losses.mean_squared_error)
```


```python
model2.fit(Xt,yt,epochs=50)
```


```python
model3 = tf.keras.Sequential([
    tf.keras.Input(shape=(11,)),
    keras.layers.Dense(units=11, activation='relu'),
    keras.layers.Dense(units=1, activation='linear')
     ])
```


```python
model3.compile(optimizer='adam',
              loss='tf.keras.losses.mean_squared_error')
model3.summary()
model3.compile(optimizer='adam',
             loss=tf.keras.losses.mean_squared_error)
```


```python
model3.fit(Xt,yt,epochs=50)
```


```python
model4 = tf.keras.Sequential([
    tf.keras.Input(shape=(11,)),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dense(units=32, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
        keras.layers.Dense(units=8, activation='relu'),
    keras.layers.Dense(units=1, activation='linear')
     ])
```


```python
model4.compile(optimizer='adam',
              loss='tf.keras.losses.mean_squared_error')
model4.summary()
model4.compile(optimizer='adam',
             loss=tf.keras.losses.mean_squared_error)
```


```python
model4.fit(Xt,yt,epochs=50)
```


```python
model5 = tf.keras.Sequential([
    tf.keras.Input(shape=(11,)),
    keras.layers.Dense(units=21, activation='relu'),
    keras.layers.Dense(units=7, activation='relu'),
    keras.layers.Dense(units=1, activation='linear')
     ])
```


```python
model5.compile(optimizer='adam',
              loss='tf.keras.losses.mean_squared_error')
model5.summary()
model5.compile(optimizer='adam',
             loss=tf.keras.losses.mean_squared_error)
```


```python
model5.fit(X_train,y_train,epochs=50)
```


```python
model5.evaluate(X_test, y_test)
```
