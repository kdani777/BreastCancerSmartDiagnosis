#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 20:23:53 2018

@author: KushDani
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataFrame = pd.read_csv('/Users/KushDani/Downloads/data.csv')

B,M = dataFrame.diagnosis.value_counts()
#print(dataFrame.head())
plt.bar(['Benign','Malignant'],[B,M], color=['green', 'red'], align='center')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.show()
#print(dataFrame.shape)


data = dataFrame.values #format DataFrame into numpy array
#print(data[0])
#print(len(data))
np.random.shuffle(data) #shuffle data
#print(data[0])

#split data into training and testing sets
trainingSet, testSet = data[:469,:], data[469:,:]
#extract corresponding labels into their own lists
trainingLabels, testLabels = trainingSet[:,1], testSet[:,1]

#modify data to for training
trainingLabels[trainingLabels == 'M'] = 1
trainingLabels[trainingLabels == 'B'] = 0
trainingLabels = trainingLabels.astype(np.float)

trainingSet[:,1] = np.nan #take diagnosis out of trainingSet
                          #to let model rely on labels and other relevant data
trainingSet = trainingSet.astype(np.float)

print(type(trainingSet)) 
print(trainingLabels[0])

#converts String to float for model fitting purposes
"""for i in range(0,len(trainingLabels) - 1):
    if(trainingLabels[i] == 'M'):
        trainingLabels[i] = 1
    else:
        trainingLabels[i] = 0"""
#modify test labels
"""for i in range(0,len(trainingSet) - 1):
    if(trainingSet[i,1] == 'M'):
        trainingSet[i,1] = 1
    else:
        trainingSet[i,1] = 0"""

print(type(trainingLabels[0]))

model = keras.Sequential()

#input layer
model.add(keras.layers.Dense(16, input_shape=(33,),
                             kernel_initializer= 'normal',
                             activation=tf.nn.leaky_relu))
#first hidden layer
model.add(keras.layers.Dropout(0.1))
#could hypothetically use regular relu activation
model.add(keras.layers.Dense(16, input_shape=(16,),
                             kernel_initializer= 'normal',
                             activation=tf.nn.leaky_relu))
model.add(keras.layers.Dropout(0.1))
#output layer
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.compile(optimizer=tf.train.AdamOptimizer(0.0009),
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

model.fit(trainingSet, trainingLabels, epochs=20, verbose=1, validation_split=0.2,
          callbacks=[keras.callbacks.EarlyStopping(monitor='acc',
                                                    patience=5)])

#model.evaluate(testSet, testLabels, verbose=0)
#^^^may need to work with larger batch size when fitting model next time
#CURRENT BEST VALIDATION ACCURACY IS 70%
