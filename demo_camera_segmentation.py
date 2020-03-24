import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Concatenate
import os
import matplotlib.pyplot as plt
from PIL import Image
import time

import os
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
from gpiozero import Button

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

def preprocess_unet(imgs):
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

filepath = 'unet.hdf5'
seg_model = get_unet()
seg_model.load_weights(filepath)


filepath = 'mobilenetv2_model.h5'
IMG_SHAPE = (224, 224, 3)
mobilev2 = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, weights='imagenet')
x = mobilev2.layers[-2].output
predictions = Dense(7, activation='softmax')(x)
model = Model(inputs=mobilev2.input, outputs=predictions)
model.load_weights(filepath)


# Set up camera constants
IM_WIDTH = 1280
IM_HEIGHT = 720
BOX_WIDTH = 360
BOX_HEIGHT = 360

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera and perform object detection.
# The camera has to be set up and used differently depending on if it's a
# Picamera or USB webcam.
label_list = ["pigmented bowen's", 'basal cell carcinoma', 'pigmented benign keratoses', 'dermatofibroma', 'melanoma', 'nevus', 'vascular']

### Picamera ###

# Initialize Picamera and grab reference to the raw capture
camera = PiCamera()
camera.resolution = (IM_WIDTH,IM_HEIGHT)
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(IM_WIDTH, IM_HEIGHT))
rawCapture.truncate(0)

infer_timestamp = 0
prediction_success_flag = False
for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    
    t1 = cv2.getTickCount()
    frame = frame1.array
    input_frame = np.array(frame)
    # input_frame = cv2.resize(src=frame, dsize=(224, 224))

    # Perform the actual detection by running the model with the image as input
    cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,120),font,1,(255,255,0),2,cv2.LINE_AA)
    cv2.putText(frame,
                "Press the button or K to capture image inside blue box",
                (30, 50),
                font,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA)
    cv2.putText(frame,
                "and do skin cancer recognition",
                (30, 80),
                font,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA)    

    
    # draw hitbox
    if time.time() - infer_timestamp > 2:
        cv2.rectangle(frame, (IM_WIDTH // 2 - BOX_WIDTH, IM_HEIGHT // 2 - BOX_HEIGHT), (IM_WIDTH // 2 + BOX_WIDTH, IM_HEIGHT // 2 + BOX_HEIGHT), (255, 0, 0), 2)
    else:
        if prediction_success_flag:
            cv2.putText(frame, 'Check detected photo.', (30, 150), font, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (IM_WIDTH // 2 - BOX_WIDTH, IM_HEIGHT // 2 - BOX_HEIGHT), (IM_WIDTH // 2 + BOX_WIDTH, IM_HEIGHT // 2 + BOX_HEIGHT), (0, 255, 0), 4)
        else:
            cv2.putText(frame, 'Nothing detected. Try again.', (30, 150), font, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (IM_WIDTH // 2 - BOX_WIDTH, IM_HEIGHT // 2 - BOX_HEIGHT), (IM_WIDTH // 2 + BOX_WIDTH, IM_HEIGHT // 2 + BOX_HEIGHT), (0, 0, 255), 4)
            
    key_event = cv2.waitKey(1)
    #if key_event == ord('k'):
    if button.is_pressed:
        infer_timestamp = time.time()
        
        cropped_frame = input_frame[IM_HEIGHT // 2 - BOX_WIDTH : IM_HEIGHT // 2 + BOX_HEIGHT, IM_WIDTH // 2 - BOX_WIDTH : IM_WIDTH // 2 + BOX_HEIGHT]
        result = model.predict(np.array([keras.applications.mobilenet.preprocess_input(np.array(cv2.resize(cropped_frame, (224, 224))))]))[0]
        print(np.argmax(result), result)

        seg_results = seg_model.predict(preprocess_unet(cropped_frame))[0]
        
        if np.max(result) > 0.4:
            for i, idx in enumerate(np.argsort(result)[:3:-1]):
                print(idx)
                print(label_list[idx])

                mask = cv2.inRange(cv2.cvtColor(prediction[0], cv2.COLOR_GRAY2RGB),
                                   (0.9, 0.9, 0.9),
                                   (1, 1, 1))
                # Create a blank 300x300 black image
                red = np.zeros((192, 240, 3), np.uint8)
                # Fill image with red color(set each pixel to red)
                red[:] = (0, 30, 0)
                cropped_frame = cv2.resize(cropped_frame, (img_cols, img_rows)) + cv2.bitwise_and(red, red, mask=mask)
                cv2.putText(cropped_frame, label_list[idx] + ' ' +  str(int(result[idx]*100)) + '%', (30, 50+i*20), font, 0.7, (255,255,255), 2, cv2.LINE_AA)
                cv2.imwrite('cropped_photo.jpg', cropped_frame)
            
            prediction_success_flag = True
            
        else:
            prediction_success_flag = False
                   
    elif key_event == ord('q'):
        break
        
    cv2.imshow('Skin Cancer Classifier', frame)
    
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1/time1

    # Press 'q' to quit

    rawCapture.truncate(0)

camera.close()

cv2.destroyAllWindows()
