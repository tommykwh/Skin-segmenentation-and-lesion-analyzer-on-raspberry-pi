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

# %matplotlib inline

raw_img = Image.open('../assets/test_image.jpg')
raw_img = np.array(raw_img)
img = tf.keras.applications.mobilenet.preprocess_input(np.array(cv2.resize(raw_img, (224, 224))))
# plt.imshow(img)

filepath = '../models/mobilenetv2_model.h5'
IMG_SHAPE = (224, 224, 3)
mobilev2 = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, weights='imagenet')
x = mobilev2.layers[-2].output
predictions = Dense(7, activation='softmax')(x)
model = Model(inputs=mobilev2.input, outputs=predictions)
model.load_weights(filepath)

result = model.predict(np.array([img]))
print(np.shape(result))
print(result)
print(np.argmax(result))


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

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]), dtype=np.uint8)
    for i, img in enumerate(imgs):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = cv2.resize(img, (IMG_SHAPE[0], IMG_SHAPE[1]), interpolation=cv2.INTER_CUBIC)
        img = np.array(img)
        imgs_p[i] = img

    imgs_p = imgs_p.astype('float32')
    mean = np.mean(imgs_p)
    std = np.std(imgs_p)
    imgs_p -= mean
    imgs_p /= std

    return imgs_p

test = preprocess(np.array([raw_img]))
prediction = seg_model.predict(test)

img = cv2.resize(raw_img, (IMG_SHAPE[0], IMG_SHAPE[1]))
plt.imshow(img)
plt.show()

mask = cv2.inRange(cv2.cvtColor(prediction[0], cv2.COLOR_GRAY2RGB),
                   (0.6, 0.6, 0.6),
                   (1, 1, 1))

# Create a blank 300x300 black image
red = np.zeros((IMG_SHAPE[0], IMG_SHAPE[1], 3), np.uint8)
# Fill image with red color(set each pixel to red)
red[:] = (0, 30, 0)

plt.imshow(img + cv2.bitwise_and(red, red, mask=mask))
plt.show()