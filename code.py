import numpy as np 
import pandas as pd 
import os
import h5py
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from matplotlib import pyplot as plt
import tensorflow as tf

# load the data
with h5py.File('../input/3d-mnist/full_dataset_vectors.h5', 'r') as dataset:
    xtrain, xtest = dataset["X_train"][:], dataset["X_test"][:]
    ytrain, ytest = dataset["y_train"][:], dataset["y_test"][:]
    
xtrain = np.array(xtrain)
xtest = np.array(xtest)

xtrain = xtrain.reshape(xtrain.shape[0], 16, 16, 16)
xtest = xtest.reshape(xtest.shape[0], 16, 16, 16)

ytrain = to_categorical(ytrain, 10)
ytest = to_categorical(ytest, 10)

print(xtrain[0].shape)

model = Sequential()

model.add(layers.Conv2D(32,(2,2),kernel_initializer='he_uniform',activation='relu',input_shape=(16,16,16),bias_initializer=Constant(0.01)))
model.add(layers.Conv2D(64,(2,2),kernel_initializer='he_uniform',activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(64,(2,2),kernel_initializer='he_uniform',activation='relu'))
model.add(layers.Conv2D(128,(2,2),kernel_initializer='he_uniform',activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Dropout(0.3))

model.add(layers.Flatten())

model.add(layers.Dense(256,'relu',kernel_initializer='he_uniform',bias_initializer=Constant(0.01)))
model.add(layers.Dropout(0.75))

model.add(layers.Dense(128,'relu',kernel_initializer='he_uniform'))
model.add(layers.Dropout(0.6))

model.add(layers.Dense(64,'relu',kernel_initializer='he_uniform'))
model.add(layers.Dropout(0.4))

model.add(layers.Dense(10,'softmax'))

model.compile(Adam(0.0005),'categorical_crossentropy',['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=4, 
                                            factor=0.5, 
                                            min_lr=1e-6)

model.fit(xtrain,ytrain,epochs=100,validation_data=(xtest,ytest),batch_size=10,verbose=2, callbacks=[EarlyStopping(patience=10),learning_rate_reduction])
