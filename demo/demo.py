import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Concatenate, Dense
import os
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2

# %matplotlib inline

raw_img = Image.open('test_image.jpg')

raw_img = np.array(raw_img)
img = tf.keras.applications.mobilenet.preprocess_input(np.array(cv2.resize(raw_img, (224, 224))))
# plt.imshow(img)

filepath = 'mobilenetv2_model.h5'
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

img_rows = 192
img_cols = 240

def get_unet():
    concat_axis = 3
    
    inputs = Input(shape=[img_rows, img_cols, 1])
    conv1 = Convolution2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Convolution2D(512, (3, 3), activation='relu', padding='same')(conv5)

#     up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    up6 = Concatenate(axis=concat_axis)([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Convolution2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv6)

#     up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    up7 = Concatenate(axis=concat_axis)([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Convolution2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv7)

#     up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    up8 = Concatenate(axis=concat_axis)([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Convolution2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv8)

#     up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    up9 = Concatenate(axis=concat_axis)([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Convolution2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Convolution2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model


filepath = 'unet.hdf5'
seg_model = get_unet()
seg_model.load_weights(filepath)

def preprocess(imgs):
    gray_image = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols, 1), dtype=np.uint8)
    resized_image = cv2.resize(gray_image, (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    imgs_p = resized_image.reshape(resized_image.shape + (1,))
    imgs_p = imgs_p.astype('float32')
    mean = np.mean(imgs_p)
    std = np.std(imgs_p)
    imgs_p -= mean
    imgs_p /= std

    return imgs_p

test = preprocess(raw_img)

prediction = seg_model.predict(np.array([test]))

img = cv2.resize(raw_img, (240, 192))
plt.imshow(img)
plt.show()

mask = cv2.inRange(cv2.cvtColor(prediction[0], cv2.COLOR_GRAY2RGB),
                   (0.9, 0.9, 0.9),
                   (1, 1, 1))
# Create a blank 300x300 black image
red = np.zeros((192, 240, 3), np.uint8)
# Fill image with red color(set each pixel to red)
red[:] = (0, 30, 0)

plt.imshow(img + cv2.bitwise_and(red, red, mask=mask))
plt.show()