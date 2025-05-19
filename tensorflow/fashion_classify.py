from tensorflow import keras
import tensorflow as tf
import numpy as np
import helper_functions as hp
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images_ = train_images/255
test_labels_ = test_labels/255

model=tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                           tf.keras.layers.Dense(128, activation='relu'),
                           tf.keras.layers.Dense(10)])


model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)

#test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions=probability_model.predict(test_images)

i = 5
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
hp.plot_image(i, predictions[i], test_labels, test_images, class_names)
plt.subplot(1,2,2)
hp.plot_value_array(i, predictions[i],  test_labels)
plt.show()

