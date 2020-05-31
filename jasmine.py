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
from tensorflow_examples.models.pix2pix import pix2pix

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

IMG_SHAPE = (224, 224, 3)
JASMINE_COL = (187,188,58)

OUTPUT_CHANNELS = 1

windowName = "JASMINE"
camera = PiCamera()
camera.rotation = 90
camera.resolution = (IM_WIDTH,IM_HEIGHT)
camera.awb_mode = 'flash'
camera.framerate = 10

freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

tap_screen = 0

y = (IM_HEIGHT - BOX_HEIGHT) // 2
x = (IM_WIDTH - BOX_WIDTH) // 2
INS_HEIGHT = 30

label_list = ["pigmented bowen's", 'basal cell carcinoma', 'pigmented benign keratoses', 'dermatofibroma', 'melanoma', 'nevus', 'vascular']
success_flag = False
analysis = []
cropped_frame = []
px1=0
px2=0
py1=0
py2=0
send_pic = False


def preprocess_unet(imgs):
    # gray_image = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
    # imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols, 1), dtype=np.uint8)
    # resized_image = cv2.resize(gray_image, (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    # imgs_p = resized_image.reshape(resized_image.shape + (1,))
    # imgs_p = imgs_p.astype('float32')
    # mean = np.mean(imgs_p)
    # std = np.std(imgs_p)
    # imgs_p -= mean
    # imgs_p /= std

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


filepath = './assets/unet.hdf5'
seg_model = get_unet()
seg_model.load_weights(filepath)


filepath = './assets/mobilenetv2_model.h5'
IMG_SHAPE = (224, 224, 3)
mobilev2 = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, weights='imagenet')
r = mobilev2.layers[-2].output
predictions = Dense(7, activation='softmax')(r)
model = Model(inputs=mobilev2.input, outputs=predictions)
model.load_weights(filepath)


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

def CallBackFunc(event, x, y, flags, param):
    global tap_screen
    global success_flag
    global send_pic
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if success_flag:
            if x < px2 and x > px1 and y < py2 and y > py1:
                send_pic = True
            else:
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
    cv2.imwrite("capture.jpg",cropped_frame)
    
    
    result = model.predict(np.array([tf.keras.applications.mobilenet.preprocess_input(np.array(cv2.resize(cropped_frame, (224, 224))))]))[0]
    seg_results = seg_model.predict(np.array(preprocess_unet([cropped_frame])))[0]
    
    if np.max(result) > 0.5:
        #timestr = time.strftime("%Y%m%d_%H%M%S")
        #cv2.imwrite("IMG_{}.png".format(timestr), cropped_frame)
        success_flag = True
        for i, idx in enumerate(np.argsort(result)[:3:-1]):
            mask = cv2.inRange(cv2.cvtColor(seg_results, cv2.COLOR_GRAY2RGB),
                                   (0.6, 0.6, 0.6),
                                   (1, 1, 1))
                # Create a blank 300x300 black image
            red = np.zeros((IMG_SHAPE[0], IMG_SHAPE[1], 3), np.uint8)
                # Fill image with red color(set each pixel to red)
            red[:] = (0, 30, 0)
            
            count = 0
            for i in range(len(mask)):
                for j in range(len(mask[i])):
                    if mask[i][j]:
                        count += 1
            frame = show_result(frame,cropped_frame,analysis)
            
            cropped_frame = cv2.resize(cropped_frame, (img_cols, img_rows)) + cv2.bitwise_and(red, red, mask=mask)
            analysis.append(label_list[idx]+': '+str(int(result[idx]*100))+'%')
        analysis.append("Approximate Size: "+str("{:.2f}".format(0.000625 * count))+"mm^2") 
    tap_screen = 0

def show_result(frame,img,analysis):
    global px1
    global px2
    global py1
    global py2
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
    px1 = x_offset
    px2 = x_offset+img.shape[1]
    
    py1 = y_offset
    py2 = y_offset+img.shape[0]
    
    output[py1:py2, px1:px2] = img
    return output
    
def read_cam(windowName):
    #t1 = cv2.getTickCount()
    global send_pic
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
            cv2.imwrite("result.jpg",frame)
            
        else:
            if time.time() - t_cap < 2:
                instruction = "Nothing Detected. Please try again."
                cv2.rectangle(frame,(x,y),(x+BOX_WIDTH, y+BOX_HEIGHT),(255,255,255),2)
            else:
                instruction = "Tap on screen to capture."
                cv2.rectangle(frame,(x,y),(x+BOX_WIDTH, y+BOX_HEIGHT),JASMINE_COL,2)
        
        if send_pic:
            os.system("blueman-sendto result.jpg capture.jpg")
            send_pic = False
            

        
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
    

