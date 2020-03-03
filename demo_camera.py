import pandas as pd
import numpy as np
import keras
from keras import backend as K
from keras.layers.core import Dense, Dropout
from keras.models import Model
import os
import matplotlib.pyplot as plt
from PIL import Image
import time

import os
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
from gpiozero import Button

mobilenet = keras.applications.mobilenet.MobileNet()
x = mobilenet.layers[-6].output
x = Dropout(0.25)(x)
predictions = Dense(7, activation='softmax')(x)
model = Model(inputs=mobilenet.input, outputs=predictions)
filepath = "model.h5"
model.load_weights(filepath)
button = Button(17)

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
        
        if np.max(result) > 0.4:
            for i, idx in enumerate(np.argsort(result)[:3:-1]):
                print(idx)
                print(label_list[idx])
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
