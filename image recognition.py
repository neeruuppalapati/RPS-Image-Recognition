import tensorflow as tf
import tensorflow.python.keras as keras
import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print shapes
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

#print first image
plt.imshow(x_train[0], cmap = 'binary')
plt.show()

# possible values
y_train[0]
print(set(y_train))

# inputting values into vector of the size of y_train
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)
print('y_train_encoded shape:', y_train_encoded.shape)
print('y_test_encoded shape:', y_test_encoded.shape)
y_train_encoded[0]
print('y_train_encoded:', y_train_encoded[0])

# creating our model with 28 * 28 nodes
x_train_reshaped = np.reshape(x_train, (60000, 784))
x_test_reshaped = np.reshape(x_test, (10000, 784))

print('x_train_reshaped shape:', x_train_reshaped.shape)
print('x_test_reshaped shape:', x_test_reshaped.shape)
print(set(x_train_reshaped[0]))

# normalizing data
x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_train_reshaped)

epsilon = 1e-10
x_train_norm = (x_train_reshaped - x_mean) / (x_std + epsilon)
x_test_norm = (x_test_reshaped - x_mean) / (x_std + epsilon)
print(set(x_train_norm[0]))

# creating the hidden layer
model = Sequential([
    # 128 nodes and 2 hidden layers was reached arbitrarily
    Dense(128, activation = 'relu', input_shape = (784,)),
    Dense(128, activation = 'relu'),
    Dense(10, activation = 'softmax')
])

# summary of the model structure
model.compile(
    optimizer = 'sgd',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model.summary()

# training the model
model.fit(x_train_norm, y_train_encoded, epochs = 3)
# run 3 times to get a better accuracy, average around 96 percent

# print accuracy
loss, accuracy = model.evaluate(x_test_norm, y_test_encoded)
print('Test set accuracy;', accuracy * 100)

# creating predictions
preds = model.predict(x_test_norm)
print("Shape of prediction:", preds.shape)

# testing model
plt.figure(figsize = (12,12))
start_index = 0

for i in range (25):
    plt.subplot(5, 5, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    pred = np.argmax(preds[start_index + i])
    gt = y_test[start_index + i]
    
    col = 'g'
    if pred != gt:
        col = 'r'
        
    plt.xlabel('i = {}, prediction = {}, gt = {}'.format(start_index + i, pred, gt), color = col)
    plt.imshow(x_test[start_index + i], cmap = 'binary')
plt.show()

# checking errors
errors = (preds != y_test_encoded)
plt.plot(errors)
plt.show()