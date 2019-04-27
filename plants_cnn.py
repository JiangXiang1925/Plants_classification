'''
****************************************************
说明：本代码执行需要提供的数据集（1.2GB）无法随附件上传，
但本人保证提交的实验结果的真实性。

作者：JiangXiang
日期：2019-04-27
*****************************************************
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.regularizers import l2,l1
from keras.utils import np_utils
import pandas as pd
from skimage.io import imread
from keras import backend as K

import matplotlib.pyplot as plt

batch_size = 128
nb_classes = 20
epochs = 50
# input image dimensions
img_rows, img_cols =64, 64
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# load the datasets
train_dataset = pd.read_csv("imagedata/plants-training-dataset.csv")
test_dataset = pd.read_csv("imagedata/plants-test-dataset.csv")
print('Datasets read.')

# this is for algorithm testing purposes, after a solution was found, apply it integrally
train_test_dataset = train_dataset.head(16000).copy()
test_test_dataset = test_dataset.head(4000).copy()
# read in the corresponding image as grayscale
train_test_dataset['img'] = train_test_dataset['path'].map(lambda x: imread(x, as_gray=False))
test_test_dataset['img'] = test_test_dataset['path'].map(lambda x: imread(x, as_gray=False))
print('Images read.')

n_samples = len(train_test_dataset['img'])
n_test = len(test_test_dataset['img'])

# put all images in an array
images = []
for i in range(0, n_samples):
    images.append(train_test_dataset['img'][i])
data_arr = np.array(images)

print(data_arr.shape)


images_test = []
for i in range(0, n_test):
    images_test.append(test_test_dataset['img'][i])

data_arr_test = np.array(images_test)
print('Arrayed Images.')


#shape data and target

if K.image_dim_ordering() == 'th':
    X_train = data_arr.reshape(data_arr.shape[0], 3, img_rows, img_cols)
    X_test = data_arr_test.reshape(data_arr_test.shape[0], 3, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = data_arr.reshape(data_arr.shape[0], img_rows, img_cols, 3)
    X_test = data_arr_test.reshape(data_arr_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

y_train = np.array(train_test_dataset['class'])
y_test = np.array(test_test_dataset['class'])
#print('y_test:',y_test)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('train samples', X_train.shape[0])
print('test samples', X_test.shape[0])

Y_train = np_utils.to_categorical(y_train,  nb_classes)
Y_test = np_utils.to_categorical(y_test,  nb_classes)
print('Y_train:',Y_train.shape)

print("**********************************************")

model = Sequential()
# 1st conv
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                        padding='same',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
# 2nd conv
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1])))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
# 3nd conv
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1])))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
# 4th conv
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1])))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
# flatten
model.add(Flatten())
# full-connection
model.add(Dense(128))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
# full-connection
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',  # adadelta
              metrics=['accuracy'])

train_log=model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(X_test, Y_test))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.show()
plt.plot(np.arange(0, epochs), train_log.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), train_log.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), train_log.history["acc"], label="train_acc")
plt.plot(np.arange(0, epochs), train_log.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on sar classifier")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig("Loss_Accuracy_cnn_{:d}e.jpg".format(epochs))


score = model.evaluate(X_test, Y_test, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
#model.summary()


