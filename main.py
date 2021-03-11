#!/usr/bin/python3

import numpy 
import tensorflow
from tensorflow.keras.datasets import mnist
import cv2
import os

from HomeworkMarking import *


def load_images_from_folder(folder):
    folder = os.getcwd() + folder
    images = []
    grays = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        # negative the picture and resize it to fit the training data
        img = cv2.resize(255-img, (28, 28))
        # cv2.imwrite("./minimize " + str(filename) + ".jpg", img)
        gray=(img/255.0)
        if gray is not None and img is not None:
            labels.append(float(os.path.splitext(filename)[0]))
            images.append(img)
            grays.append(gray)
    return images, grays, numpy.array(labels)
    
def dataPreprocessing():
    (dataTrain, labelTrain), (dataTest, labelTest) = mnist.load_data()
    # normalize datas
    maxVal = numpy.max(dataTrain)
    dataTrain = (dataTrain / maxVal).reshape(60000, 28, 28, 1)
    dataTest = (dataTest / maxVal).reshape(10000, 28, 28, 1)
    return dataTrain, labelTrain, dataTest, labelTest

def compareResult(predict, images):
    for pred, img in zip(predict, images):
        print(pred)
        cv2.imshow('output', img)
        cv2.waitKey(0)

def displayContour():
    img = cv2.imread('test/testOperation.jpg')
    # img = cv2.imread('test/childrenDigits/0.png')
    imgray = 255 - img
    imgray = cv2.cvtColor(imgray, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    cv2.drawContours(img, contours, 0, (0, 255, 0), 3)
    cv2.imshow('test', img)
    cv2.waitKey(0)

def main():
    folder = "/test/childrenDigits/"
    dataTrain, labelTrain, dataTest, labelTest = dataPreprocessing()
    sourcesImages, dataTest, labelTest = load_images_from_folder(folder)
    dataTest=numpy.array(dataTest).reshape(4,28,28,1)

    hm = HomeworkMarking("logs/checkpoints/cp1.ckpt")
    hm.evaluate(dataTest, labelTest)
    predict = hm.predict(dataTest)

    # compareResult(predict, sourcesImages)

    # print(test)
    # print(dataTest)
    # print(labelTrain)
    # print(dataTest.shape, dataTest.dtype)
    # print(labelTest.shape, labelTest.dtype)
    return 0

if __name__ == "__main__":
    main()