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

def convert_to_tflite(model, filename):
  # Convert the model.
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()

  # Save the TF Lite model.
  with tf.io.gfile.GFile(filename, 'wb') as f:
    f.write(tflite_model)

def test_converted_tflite(filename):
  # convert_to_tflite(model, filename)
  interpreter = tf.lite.Interpreter(model_path=filename)
  interpreter.allocate_tensors()

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  print(input_details)
  output_details = interpreter.get_output_details()
  print(output_details)

  # Test the TensorFlow Lite model on random input data.
  input_shape = input_details[0]['shape']
  input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
  interpreter.set_tensor(input_details[0]['index'], input_data)

  interpreter.invoke()

  # The function `get_tensor()` returns a copy of the tensor data.
  # Use `tensor()` in order to get a pointer to the tensor.
  # tflite_results = interpreter.get_tensor(output_details[0]['index'])

filepath = '../models/mobilenetv2_model.h5'
IMG_SHAPE = (224, 224, 3)
mobilev2 = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, weights='imagenet')
x = mobilev2.layers[-2].output
predictions = Dense(7, activation='softmax')(x)
model = Model(inputs=mobilev2.input, outputs=predictions)
model.load_weights(filepath)
filename = 'mobilenet_v2.tflite'

convert_to_tflite(model, filename)
test_converted_tflite(filename)

def unet_model(recg_model, output_channels):
    # Create the base model from the pre-trained model MobileNet V2
    # mobilev2 = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)
    # x = mobilev2.layers[-2].output
    # predictions = Dense(7, activation='softmax')(x)
    # base_model = Model(inputs=mobilev2.input, outputs=predictions)
    # base_model.load_weights(filepath)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    layers = [recg_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=recg_model.input, outputs=layers)
    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]

    inputs = tf.keras.layers.Input(shape=IMG_SHAPE)
    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=1,
        kernel_size=3,
        strides=2,
        padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

OUTPUT_CHANNELS = 1
filepath = '../models/tf_unet.hdf5'
seg_model = unet_model(model, OUTPUT_CHANNELS)
seg_model.load_weights(filepath)
filename = 'unet.tflite'

convert_to_tflite(seg_model, filename)
test_converted_tflite(filename)