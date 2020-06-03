import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2


def test(model, imgs):
    accuracy = 0
    for i, img in enumerate(imgs):
        img = np.array([tf.keras.applications.mobilenet.preprocess_input(np.array(cv2.resize(img, (224, 224))))])
        prediction =  model.predict(img)

        if np.argmax(result) == y_test[i]:
            accuracy += 1

    accuracy /= len(imgs)
    return accuracy

def test_tflite(model, imgs):
    accuracy = 0
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    for i, img in enumerate(imgs):
        img = np.array([tf.keras.applications.mobilenet.preprocess_input(np.array(cv2.resize(img, (224, 224))))])
        model.set_tensor(input_details[0]['index'], img)
        model.invoke()

        result = model.get_tensor(output_details[0]['index'])

        if np.argmax(result) == y_test[i]:
            accuracy += 1

    accuracy /= len(imgs)
    return accuracy

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

filepath = '../models/tf_unet.hdf5'
seg_model = unet_model(model, OUTPUT_CHANNELS)
seg_model.load_weights(filepath)

def test_seg(model, imgs):
    accuracy = 0
    for i, img in enumerate(imgs):
        area = 0
        prediction = model.predict(img)
        confidence = 0.9
        _, mask_to_show = cv2.threshold(prediction, confidence, 1, cv2.THRESH_BINARY)
        mask = cv2.inRange(mask_to_show,
                        (confidence),
                        (1))

        # Create a blank 300x300 black image
        red = np.zeros((224, 224, 3), np.uint8)
        # Fill image with red color(set each pixel to red)
        red[:] = (0, 50, 0)
        for m in range(len(mask)):
            for n in range(len(mask[i])):
                if mask[m][n] == y_test[i][m][n]:
                    area += 1
        
        area /= 224 * 224
    
    accuracy /= len(imgs)
    return accuracy


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

def test_seg(model, imgs):
    accuracy = 0
    for i, img in enumerate(imgs):
        area = 0
        prediction = model.predict(img)
        confidence = 0.6
        _, mask_to_show = cv2.threshold(prediction, confidence, 1, cv2.THRESH_BINARY)
        mask = cv2.inRange(mask_to_show,
                        (confidence),
                        (1))

        # Create a blank 300x300 black image
        red = np.zeros((224, 224, 3), np.uint8)
        # Fill image with red color(set each pixel to red)
        red[:] = (0, 50, 0)
        for m in range(len(mask)):
            for n in range(len(mask[i])):
                if mask[m][n] == y_test[i][m][n]:
                    area += 1
        
        area /= 224 * 224
    
    accuracy /= len(imgs)
    return accuracy

def test_seg_tfilte(model, imgs):
    accuracy = 0
    for i, img in enumerate(imgs):
        area = 0
        # Get input and output tensors.
        input_details = seg_model.get_input_details()
        output_details = seg_model.get_output_details()

        # Test the TensorFlow Lite model on random input data.
        seg_model.set_tensor(input_details[0]['index'], img)
        seg_model.invoke()

        prediction = seg_model.get_tensor(output_details[0]['index'])
        confidence = 0.6
        _, mask_to_show = cv2.threshold(prediction, confidence, 1, cv2.THRESH_BINARY)
        mask = cv2.inRange(mask_to_show,
                        (confidence),
                        (1))

        # Create a blank 300x300 black image
        red = np.zeros((224, 224, 3), np.uint8)
        # Fill image with red color(set each pixel to red)
        red[:] = (0, 50, 0)
        for m in range(len(mask)):
            for n in range(len(mask[i])):
                if mask[m][n] == y_test[i][m][n]:
                    area += 1
        
        area /= 224 * 224
    
    accuracy /= len(imgs)
    return accuracy
        



