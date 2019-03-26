import numpy as np
import pickle
import scipy.io as sio
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.models import load_model

x_train, y_train = pickle.load(open('file_train_set.npy','rb')), pickle.load(open('file_train_set_labels.npy','rb'))
x_valid, y_valid = pickle.load(open('file_valid_set.npy','rb')), pickle.load(open('file_valid_set_labels.npy','rb'))
x_test, y_test = pickle.load(open('file_test_set.npy','rb')), pickle.load(open('file_test_set_labels.npy','rb'))

model = Sequential()
print(x_train[0].shape)
model.add(Flatten(input_shape=(10,21)))
model.add(Dense(210))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))

opt = Adam()
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

print("Training")
history = model.fit(x_train, y_train, batch_size=32, epochs=1, validation_data=(x_valid, y_valid))

model3.save('model_file.h5')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
