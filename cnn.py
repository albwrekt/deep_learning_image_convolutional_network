#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 21:46:52 2020

Description: This is a convolutional network to recognize images of pets.

@author: albwrekt
"""

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

# Part 1 -> Data Preprocessing Images

# Apply transformations to all images of training set, not test set
# Transformations are simple, geometric including zooms, flips to get them augmented. This is called image augmentation
train_datagen = ImageDataGenerator(
    # This applies feature scaling to every pixel by dividing every pixel by 255 to normalize
    rescale=1./255,
    #
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )

training_set = train_datagen.flow_from_directory(
    #location of training set
    './training_set',
    #can greatly affect speed
    target_size=(64,64),
    #How many images in each batch, 32 is classic value
    batch_size=32,
    # two output neurons
    class_mode='binary'
    )

# Preprocessing the training set
# We don't want to touch test images. Must stay in tact, and must be rescaled.
# The images will have rescaled pixels. Transform is only used on training set. Both are feature scaled.
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    './test_set',
    #must be same size as training set
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
    )


# Part 2 ->  Building the CNN

#ANN as sequence of layers rather than computational graph
cnn = tf.keras.models.Sequential()

# Step 1 -> Convolution

# Important parameters: fileters, kernel size, activation, input_shape
# Filters -> Number of feature detectors
# Kernel Size -> Number of rows and columns
# Activation -> This is used to activate the ReLU layer
# Input Shape -> Size of reshaped images, # of values representing color
cnn.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=3,
    activation='relu',
    # resized image shapes, and RGB color representation
    input_shape=[64,64,3]
    ))

# Step 2 -> Pooling

# This will have max pooling.
# Add pooling layers. Padding values are automatically added
# Pool Size -> Frame of pool iterating over image. Length of square size. 2 is recommended for max pooling
# Strides -> How many pixels is the frame shifted to right each time
cnn.add(tf.keras.layers.MaxPool2D(
    pool_size=2,
    strides=2
    ))

#Repeat Step 1 -> Convolution

#input shape is not added to any layer other than original layer. Size is assumed.
cnn.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=3,
    activation='relu'
    ))

# Repeat Step 2 -> Pooling

cnn.add(tf.keras.layers.MaxPool2D(
    pool_size=2,
    strides=2
    ))

# Step 3 -> Flattening

# This compresses the results down to a flattened layer for ANN processing.
cnn.add(tf.keras.layers.Flatten())


# Step 4 -> Full Connection

#Full implementation of ANN fed by flattened data

# Hidden Layers
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 -> Output Layer

# Output Layers
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 -> Training the CNN

# Connect to optimizer, loss function, and metrics
# Adam optimizer includes stochastic gradient descent. 
# Binary classification uses binary_crossentropy
# Accuracy as a metric to evaluate the accuracy of the model
# This training is different because training and testing are happening together.

#Compile the network
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train and Test simultaneously
# validation data is the test set to validate against
cnn.fit(x=training_set, validation_data=test_set, epochs=25)


# Part 4 -> Making a single prediction
test_image_one = image.load_img(
    './single_prediction/cat_or_dog_1.jpg',
    target_size=(64,64)
    )

test_image_two = image.load_img(
    './single_prediction/cat_or_dog_2.jpg',
    target_size=(64,64)
    )

# Convert the image to a 2D array for processing
test_image_one = image.img_to_array(test_image_one)
test_image_two = image.img_to_array(test_image_two)

# Predict method must be used on same format in training
# this CNN was trained on batches, which must be replicated.
test_image_one = np.expand_dims(test_image_one, axis=0)
test_image_two = np.expand_dims(test_image_two, axis=0)

result_one = cnn.predict(test_image_one)
result_two = cnn.predict(test_image_two)

print(training_set.class_indices)

# To get batch prediction, access the batch then parameter
# Use the encoding to access the prediction value.
if result_one[0][0] == 1:
    prediction_one = 'dog'
else:
    prediction_one = 'cat'
    
if result_two[0][0] == 1:
    prediction_two = 'dog'
else:
    prediction_two = 'cat'
    
    
# Print out final results
print("The result of pic 1: " + prediction_one)
print("The result of prediction 2: " + prediction_two)