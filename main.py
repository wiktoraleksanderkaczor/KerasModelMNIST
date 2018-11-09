import _pickle as cPickle
import gzip
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import np_utils

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
num_pix = X_train.shape[1] * X_train.shape[2] #28*28
X_train = X_train.reshape(X_train.shape[0], num_pix).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pix).astype('float32')

X_train = X_train / 255 #Turn black and white
X_test = X_test / 255

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]

model = Sequential()
model.add(Dense(units=num_pix, activation='relu', input_dim=num_pix))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=3, batch_size=30, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
