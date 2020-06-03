from flask import Flask, render_template, request, url_for
import tensorflow as tf
import numpy as np
import os
import json
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os

IMG_SHAPE = (224, 224, 3)
OUTPUT_CHANNELS = 1
label_list = ["pigmented bowen's", 'basal cell carcinoma', 'pigmented benign keratoses', 'dermatofibroma', 'melanoma', 'nevus', 'vascular']
classification = ['malignant', 'indeterminate', 'benign']
app = Flask(__name__, template_folder=os.path.abspath("../electron"))
filepath = 'static/mobilenet_v2.tflite'
model = tf.lite.Interpreter(model_path=filepath)
model.allocate_tensors()

filepath = 'static/unet.tflite'
seg_model = tf.lite.Interpreter(model_path=filepath)
seg_model.allocate_tensors()

def predict(model, raw_img):
    # Get input and output tensors.
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    img = np.array([tf.keras.applications.mobilenet.preprocess_input(np.array(cv2.resize(raw_img, (224, 224))))])
    model.set_tensor(input_details[0]['index'], img)
    model.invoke()

    result = model.get_tensor(output_details[0]['index'])[0]

    acc = []
    classes = []
    for i, idx in enumerate(np.argsort(result)[:3:-1]):
        acc.append(result[idx])
        classes.append(label_list[idx])

    print(acc)
    print(classes)
    return acc, classes

def predict_seg(model, raw_img):
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

    # Get input and output tensors.
    input_details = seg_model.get_input_details()
    output_details = seg_model.get_output_details()

    # Test the TensorFlow Lite model on random input data.
    test = preprocess(np.array([raw_img]))
    seg_model.set_tensor(input_details[0]['index'], test)
    seg_model.invoke()

    prediction = seg_model.get_tensor(output_details[0]['index'])

    img = cv2.resize(raw_img, (IMG_SHAPE[0], IMG_SHAPE[1]))

    confidence = 0.5
    mask = cv2.inRange(cv2.cvtColor(prediction[0], cv2.COLOR_GRAY2RGB),
                    (confidence, confidence, confidence),
                    (1, 1, 1))

    # Create a blank 300x300 black image
    red = np.zeros((IMG_SHAPE[0], IMG_SHAPE[1], 3), np.uint8)
    # Fill image with red color(set each pixel to red)
    red[:] = (255, 255, 255)
    red = cv2.bitwise_and(red, red, mask=mask)

    alpha = 0.6
    seg_img = cv2.addWeighted(img, alpha, red, 1 - alpha, 0)
    cv2.imwrite('static/seg_img.jpg', cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)) 

@app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method == "GET":
        return render_template("RESULT.html")

    if request.method == "POST":
        path = request.form["path"]
        img = Image.open(path)
        img = np.array(img)
        accuracies, classes = predict(model, img)
        predict_seg(seg_model, img)

        return render_template("RESULT.html",
                                preds=accuracies,
                                classes=json.dumps(classes))

app.run()


