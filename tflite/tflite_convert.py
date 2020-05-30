import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Concatenate, Dense
from tensorflow_examples.models.pix2pix import pix2pix
import os
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2

# Create a simple Keras model.
filepath = '../models/mobilenetv2_model.h5'
IMG_SHAPE = (224, 224, 3)
mobilev2 = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, weights='imagenet')
x = mobilev2.layers[-2].output
predictions = Dense(7, activation='softmax')(x)
model = Model(inputs=mobilev2.input, outputs=predictions)
model.load_weights(filepath)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TF Lite model.
with tf.gfile.GFile('mobilenetv2.tflite', 'wb') as f:
  f.write(tflite_model)