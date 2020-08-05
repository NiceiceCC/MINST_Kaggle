# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 21:11:04 2020

@author: yangx
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 21:24:36 2020

@author: yangx
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# import seaborn as sns
# %matplotlib inline

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,BatchNormalization,GlobalAveragePooling2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint

# Load the data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

Y_train = train["label"]

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 

# free some space
del train 

X_train.isnull().any().describe()
test.isnull().any().describe()

X_train = X_train / 255.0
test = test / 255.0

X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

Y_train = to_categorical(Y_train, num_classes = 10)

random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
#%%
def buildFCN():
    model = Sequential()
    
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu', input_shape = (28,28,1)))
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    # model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
    #                   activation ='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2),strides = (2,2)))
    # model.add(Dropout(0.25))
    
    
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    # model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
    #                  activation ='relu'))
    # model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     # activation ='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    
    # model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 
    #                  activation ='relu'))
    # model.add(Conv2D(filters = 128, kernel_size = (1,1),padding = 'Same', 
    #                  activation ='relu'))
    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 
                      activation ='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(filters = 256, kernel_size = (1,1),padding = 'Same', 
                       activation ='relu'))
    model.add(Conv2D(filters = 10, kernel_size = (1,1),padding = 'Same', 
                       activation ='relu'))
    model.add(GlobalAveragePooling2D())
    return model

def buildNoDropout():
    model = Sequential()
    
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu', input_shape = (28,28,1)))
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                      activation ='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2),strides = (2,2)))
    # model.add(Dropout(0.25))
    
    
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                      activation ='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 
                      activation ='relu'))
    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 
                      activation ='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))    
    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = "softmax"))
    return model
model = buildFCN()
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
model.summary()
#%%
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)
checkpoint = ModelCheckpoint(filepath='FCN_Full.hd5', mode='max', monitor='val_accuracy', verbose=2, save_best_only=True)
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1, 

        # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)
#%%
# Fit the model
epochs = 30 # Turn epochs to 30 to get 0.9967 accuracy
batch_size  = 80

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size), 
                              epochs = epochs, validation_data = (X_val,Y_val), 
                              verbose = 2, steps_per_epoch=X_train.shape[0]
                              , callbacks=[ checkpoint])

#%%
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.show()
#%%
model = buildNoDropout()
model.load_weights('FCN99643.hd5')
test_label = model.predict(test)
test_label = np.argmax(test_label, axis = 1)
df = pd.DataFrame([[i+1,l] for i,l in enumerate(test_label)],columns = ['ImageId','Label'])
df.to_csv('submit.csv', index = False)
