import sys
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Concatenate, Dense

import os
import matplotlib.pyplot as plt
from PIL import Image
import time



#IM_WIDTH = 1280
#IM_HEIGHT = 720

IM_WIDTH = 480
IM_HEIGHT = 320

BOX_WIDTH = 260
BOX_HEIGHT = 260

img_rows = 192
img_cols = 240

JASMINE_COL = (187,188,58)

windowName = "JASMINE"
camera = PiCamera()
camera.rotation = 90
camera.resolution = (IM_WIDTH,IM_HEIGHT)
camera.awb_mode = 'flash'
camera.framerate = 10

freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

tap_screen = 0

y = (IM_HEIGHT - BOX_HEIGHT)//2
x = (IM_WIDTH - BOX_WIDTH)//2
INS_HEIGHT = 30

label_list = ["pigmented bowen's", 'basal cell carcinoma', 'pigmented benign keratoses', 'dermatofibroma', 'melanoma', 'nevus', 'vascular']
success_flag = False
analysis = []
cropped_frame = []

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
    gray_image = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
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
r = mobilev2.layers[-2].output
predictions = Dense(7, activation='softmax')(r)
model = Model(inputs=mobilev2.input, outputs=predictions)
model.load_weights(filepath)

def CallBackFunc(event, x, y, flags, param):
    global tap_screen
    global success_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        if success_flag:
            success_flag = False
        else:
            tap_screen = 1
   
def open_window(windowName, width, height):
    cv2.namedWindow(windowName, cv2.WINDOW_KEEPRATIO)
    #cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, width, height)
    cv2.moveWindow(windowName, 0, 0)
    cv2.setWindowTitle(windowName, "JASMINE")

def take_capture(frame):
    global tap_screen
    global success_flag
    global analysis
    global cropped_frame
    success_flag = False
    analysis = []
    cropped_frame = []
    input_frame = np.array(frame)
    cropped_frame = input_frame[y: y+BOX_HEIGHT, x: x+ BOX_WIDTH]
    result = model.predict(np.array([tf.keras.applications.mobilenet.preprocess_input(np.array(cv2.resize(cropped_frame, (224, 224))))]))[0]
    seg_results = seg_model.predict(np.array([preprocess_unet(cropped_frame)]))[0]
    
    if np.max(result) > 0.5:
        #timestr = time.strftime("%Y%m%d_%H%M%S")
        #cv2.imwrite("IMG_{}.png".format(timestr), cropped_frame)
        success_flag = True
        for i, idx in enumerate(np.argsort(result)[:3:-1]):
            mask = cv2.inRange(cv2.cvtColor(seg_results, cv2.COLOR_GRAY2RGB),
                                   (0.9, 0.9, 0.9),
                                   (1, 1, 1))
                # Create a blank 300x300 black image
            red = np.zeros((192, 240, 3), np.uint8)
                # Fill image with red color(set each pixel to red)
            red[:] = (0, 30, 0)
            
            count = 0
            for i in range(len(mask)):
                for j in range(len(mask[i])):
                    if mask[i][j]:
                        count += 1
            
            cropped_frame = cv2.resize(cropped_frame, (img_cols, img_rows)) + cv2.bitwise_and(red, red, mask=mask)
            analysis.append(label_list[idx]+': '+str(int(result[idx]*100))+'%')
        analysis.append("Approximate Size: "+str("{:.2f}".format(0.000625 * count))+"mm^2") 
    tap_screen = 0

def show_result(frame,img,analysis):
    alpha = 0.6
    overlay = frame.copy()
    output = frame.copy()
    
    PIC_SIZE = 180
    
    p1_x = 0
    p1_y = INS_HEIGHT // 2
    p2_x = IM_WIDTH
    p2_y = IM_HEIGHT - INS_HEIGHT // 2
    
    x_offset = 10
    y_offset = (IM_HEIGHT - INS_HEIGHT - PIC_SIZE) // 2 + INS_HEIGHT // 2
    
    cv2.rectangle(overlay, (p1_x, p1_y), (p2_x, p2_y),JASMINE_COL, -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha,0, output)
    
    for i, item in enumerate(analysis):
        cv2.putText(output,item,(p1_x+PIC_SIZE+x_offset*2, p1_y+y_offset+i*50),font,0.5,(255, 255, 255),1,cv2.LINE_AA)
        
    img = cv2.resize(img, (PIC_SIZE, PIC_SIZE))    
    output[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img
    return output
    
def read_cam(windowName):
    #t1 = cv2.getTickCount()
    t_cap = 0
    
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH, IM_HEIGHT))
    rawCapture.truncate(0)
    for f in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
        key = cv2.waitKey(10)
        if key == 27:
            break
        frame = f.array
        
        
        if tap_screen == 1:
            take_capture(frame)
            t_cap = time.time()
            
        if success_flag:
            instruction = "Tap on screen to close analysis."
            frame = show_result(frame,cropped_frame,analysis)
        else:
            if time.time() - t_cap < 2:
                instruction = "Nothing Detected. Please try again."
                cv2.rectangle(frame,(x,y),(x+BOX_WIDTH, y+BOX_HEIGHT),(255,255,255),2)
            else:
                instruction = "Tap on screen to capture."
                cv2.rectangle(frame,(x,y),(x+BOX_WIDTH, y+BOX_HEIGHT),JASMINE_COL,2)
        
        
            

        
        M = np.float32([[1,0,0],[0,1,INS_HEIGHT//2]]) #MOVE DOWN
        frame = cv2.warpAffine(frame,M,(IM_WIDTH,IM_HEIGHT))
        
        #cv2.rectangle(dst,(0,0),(IM_WIDTH, 100),(0,0,255),-1)
        
        x_off = 20
        y_off = INS_HEIGHT - 8
    
        cv2.rectangle(frame, (0, 0), (IM_WIDTH, INS_HEIGHT),JASMINE_COL, -1)
        cv2.putText(frame,instruction,(x_off, y_off),font,0.8,(255, 255, 255),1,cv2.LINE_AA)
        
        cv2.imshow(windowName, frame)        
        rawCapture.truncate(0)
        cv2.setWindowProperty(windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    open_window(windowName, IM_WIDTH, IM_HEIGHT)
    cv2.setMouseCallback(windowName, CallBackFunc)
    read_cam(windowName)
    

