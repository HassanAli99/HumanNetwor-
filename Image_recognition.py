#Importing Libraries
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#Loading the dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


#Data Preprocessing
def data_preparation(x_train, y_train, x_test, y_test):

  #Reshape the data
  print("Shape of the train and test data before transforming is {}, {}".format(x_train.shape, x_test.shape))
  x_train_reshaped = x_train.reshape(60000, 784)
  x_test_reshaped = x_test.reshape(10000, 784)

  print("Shape of the train and test data after reshaping is {}, {}".format(x_train_reshaped.shape, x_test_reshaped.shape))


  #Rescale the x_train and x_test to values between zero and one by dividing each variable by 255
  x_train_scaled = x_train_reshaped/256
  x_test_scaled = x_test_reshaped/256


  #Convert the train and testset labels to two new variables, called y_train and y_test.
  lebels = 10
  y_train = keras.utils.to_categorical(y_train, lebels)
  y_test = keras.utils.to_categorical(y_test, lebels)

  #print("Min and max of the train data after scaling: {}, {}".format(min(x_train_scaled), max(x_test_scaled)))
  return x_train_scaled, y_train, x_test_scaled, y_test


x_train, y_train, x_test, y_test = data_preparation(x_train, y_train, x_test, y_test)

## Model definition:

model = keras.Sequential()
model.add(keras.layers.Dense(256, input_shape=(784,)))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=128, epochs=12, verbose=1, validation_split=0.2)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Loss')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Model Accuracy is {} and the loss is  {}".format(accuracy, loss))