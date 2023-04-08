import tensorflow as tf
import numpy as np
import pandas as pd

df = pd.read_excel("AirQualityUCI.xlsx")

df['Date']=df['Date'].dt.month
df['Time']=df['Time'].apply(lambda x: int(x.strftime('%H')))

features = df.iloc[:, :-1]
labels = df.iloc[:, -1]

labels = pd.get_dummies(labels).values

train_data = features[int(features.shape[0] * 0.8), :]
train_labels = labels[int(labels.shape[0] * 0.8), ]
test_data = features[int(features.shape[0] * 0.2):, :]
test_labels = labels[int(labels.shape[0] * 0.2):, ]

test_error = []

for i in range(5):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(28 * 28, )))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

    test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=0)
    test_error.append(1 - test_acc)

for i, error in enumerate(test_error):
    print('Architecture {} Test Error: {:.4f}'.format(i + 1, error))

best_architecture = np.argmin(test_error) + 1
print('Best Architecture:', best_architecture)
