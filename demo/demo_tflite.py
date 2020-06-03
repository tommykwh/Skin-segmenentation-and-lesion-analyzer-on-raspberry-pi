import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2

IMG_SHAPE = (224, 224, 3)
raw_img = Image.open('../assets/test_image.jpg')
raw_img = np.array(raw_img)
img = tf.keras.applications.mobilenet.preprocess_input(np.array(cv2.resize(raw_img, (224, 224))))
# plt.imshow(img)

filepath = '../tflite/mobilenet_v2.tflite'

model = tf.lite.Interpreter(model_path=filepath)
model.allocate_tensors()

# Test the TensorFlow Lite model on random input data.
def predict(model, img):
    # Get input and output tensors.
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    img = np.array([tf.keras.applications.mobilenet.preprocess_input(np.array(cv2.resize(raw_img, (224, 224))))])
    model.set_tensor(input_details[0]['index'], img)
    model.invoke()

    result = model.get_tensor(output_details[0]['index'])
    print(np.shape(result))
    print(result)
    print(np.argmax(result))

predict(model, raw_img)
predict(model, raw_img)

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

OUTPUT_CHANNELS = 1
filepath = '../tflite/unet.tflite'
seg_model = tf.lite.Interpreter(model_path=filepath)
seg_model.allocate_tensors()


def predict_seg(model, img):
    # Get input and output tensors.
    input_details = seg_model.get_input_details()
    output_details = seg_model.get_output_details()

    # Test the TensorFlow Lite model on random input data.
    test = preprocess(np.array([raw_img]))
    seg_model.set_tensor(input_details[0]['index'], test)
    seg_model.invoke()

    prediction = seg_model.get_tensor(output_details[0]['index'])

    img = cv2.resize(raw_img, (IMG_SHAPE[0], IMG_SHAPE[1]))
    # plt.imshow(img)
    # plt.show()

    confidence = 0.6
    mask = cv2.inRange(cv2.cvtColor(prediction[0], cv2.COLOR_GRAY2RGB),
                    (confidence),
                    (1))

    # Create a blank 300x300 black image
    red = np.zeros((IMG_SHAPE[0], IMG_SHAPE[1], 3), np.uint8)
    # Fill image with red color(set each pixel to red)
    red[:] = (255, 255, 255)
    red = cv2.bitwise_and(red, red, mask=mask)

    # plt.imshow(img + cv2.bitwise_and(red, red, mask=mask)
    alpha = 0.6
    img = cv2.addWeighted(img, alpha, red, 1 - alpha, 0)
    cv2.imwrite('seg_img.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) 
       

predict_seg(seg_model, raw_img)