from tensorflow import keras
import numpy as np


model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

train_data = []
labels =[]

for i in range(10):
    train_data.append(i)

##creating an array
for i in train_data:
    labels.append(3 * i + 1)


model.fit(np.array(train_data, dtype=float), np.array(labels, dtype=float), epochs=150)

print(model.predict(np.array([-1], dtype=float)))
print(labels)

