import tensorflow as tf
import cv2 as cv
import numpy as np
from tensorflow.keras.preprocessing import image

modelPath = "idenprof_models/final_model.h5"

model = tf.keras.models.load_model(modelPath)

def processImage(imagePath, targetSize=(100,100)):
    # gscImg = cv.cvtColor(imagePath, cv.COLOR_RGB2GRAY)
    loadImage = image.load_img(path=imagePath, target_size=targetSize, color_mode="rgba")
    imgArray = image.img_to_array(loadImage)
    imgArray = np.expand_dims(imgArray, axis=0)
    imgArray = imgArray/255.0 #rescaling the image back
    return imgArray


def predictImage(modelName, imgPath):
    img_array = processImage(imgPath)
    print(img_array)
    prediction = modelName.predict(img_array)
    return prediction


prediction = predictImage(modelName=model, imgPath="D:/Data Science/Machine Learning/Identity Proof/idenprof/test/pilot/pilot-176.jpg")

print(prediction)
