import numpy as np
from sklearn.datasets import fetch_openml
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist  #returns 2 tuples
(X_train, y_train),(X_test, y_test) = mnist.load_data()
X_train.shape

y_train.shape



some_digit = X_train[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")
plt.savefig("digit_plot.png",format='png', dpi=200)
plt.show()

y_train[0]


#normalize -> manual scaling (xmax-min)/(max-min)
X_train = X_train/255.0
X_test = X_test/255.0

#train the model compraised by 28x28 input neurons followed by 2 dense layers 
model = keras.models.Sequential()
#series of uncorrelated neurons (starting point). 
model.add(keras.layers.Flatten(input_shape=[28,28]))
#each neuron is affected by all the neurons of the previous layer
#relu is the best activation function for gradient descend
model.add(keras.layers.Dense(100, activation = "relu"))
model.add(keras.layers.Dense(50, activation = "relu"))
#generally transforms the outputs into 0,1 so we can classify them more easily.As an outcome we choose the  digit(neuron) 
#with the highest probability
model.add(keras.layers.Dense(10, activation = "softmax"))


model.summary()
keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#compile
#sparse categorical crossentropy -> it is used similarly to mean squared error, it is our loss function, and it is a benchmark for classifying the result into a category (class).
model.compile(loss = 'sparse_categorical_crossentropy', 
              optimizer = "sgd",
              metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=13, validation_split=0.1)

#plot
pd.DataFrame(history.history).plot(figsize = (8,5))
plt.grid()
plt.gca().set_ylim(0,1)
plt.xlabel('Epochs')
plt.ylabel('Loss & Accuracy')
plt.savefig("results_plot.png",format='png', dpi=200)
plt.show()

#evaluate
model.evaluate(X_test, y_test)

#check prediction on test set
X_check = X_test[10:20]
y_probability = model.predict(X_check)
y_probability.round(2)

#matches the highest probability of the prediction (y_probability ) with the equivalent class
y_predict = model.predict_classes(X_check)
y_predict

y_original = y_test[10:20]
y_original


