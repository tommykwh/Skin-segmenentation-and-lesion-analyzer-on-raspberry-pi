import pandas as pd
import numpy as np
'''
import keras
from keras import backend as K
from keras.layers.core import Dense, Dropout
from keras.models import Model
'''
import os
import matplotlib.pyplot as plt
from PIL import Image

import os
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
'''
mobilenet = keras.applications.mobilenet.MobileNet()
x = mobilenet.layers[-6].output
x = Dropout(0.25)(x)
predictions = Dense(7, activation='softmax')(x)
model = Model(inputs=mobilenet.input, outputs=predictions)
filepath = "model.h5"
model.load_weights(filepath)
'''
# Set up camera constants
# IM_WIDTH = 1280
# IM_HEIGHT = 720
IM_WIDTH = 1280
IM_HEIGHT = 720

# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)

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

for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

    t1 = cv2.getTickCount()
    frame = frame1.array
    input_frame = cv2.resize(src=frame, dsize=(224, 224))

    # Perform the actual detection by running the model with the image as input
    '''
    result = model.predict(np.array([keras.applications.mobilenet.preprocess_input(np.array(input_frame))]))
    if 0.8 > np.max(result) and np.max(result) > 0.4:
        print(np.argmax(result))
        cv2.putText(frame, label_list[np.argmax(result)], (30,50), font, 1, (0,255,0), 2, cv2.LINE_AA)
    '''
    cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,100),font,1,(255,255,0),2,cv2.LINE_AA)
    cv2.putText(frame,
                "Press the button or K to capture image inside blue box and do skin cancer recognition",
                (30, 50),
                font,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA)    

    
    # draw hitbox 
    cv2.rectangle(frame, (IM_WIDTH // 2 - 112, IM_HEIGHT // 2 - 112), (IM_WIDTH // 2 + 112, IM_HEIGHT // 2 + 112), (255,0,0), 2)
    cv2.imshow('Skin Cancer Classifier', frame)
    
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1/time1

    # Press 'q' to quit
    
    key_event = cv2.waitKey(1)
    if key_event == ord('k'):
        cropped_frame = frame[IM_HEIGHT // 2 - 112:IM_HEIGHT // 2 + 112, IM_WIDTH // 2 - 112:IM_WIDTH // 2 + 112]
        cv2.imwrite('cropped_photo.jpg', cropped_frame)
        
    elif key_event == ord('q'):
        break

    #if cv2.waitKey(1) == ord('b'):



    rawCapture.truncate(0)

camera.close()

cv2.destroyAllWindows()

