#!/usr/bin/python3

import numpy 
import tensorflow
from tensorflow.keras.datasets import mnist
import cv2
import os

from HomeworkMarking import *


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize( img,(28, 28))
        img=(img/255.0)
        if img is not None:
            images.append(img)
    return images

folder = r'C:\\Users\Madalina Aldea\Desktop\JU\Machine learning\project1\homeworkMarking\test'
    
def dataPreprocessing():
    (dataTrain, labelTrain), (dataTest, labelTest) = mnist.load_data()
    # normalize datas
    maxVal = numpy.max(dataTrain)
    dataTrain = (dataTrain / maxVal).reshape(60000, 28, 28, 1)
    dataTest = (dataTest / maxVal).reshape(10000, 28, 28, 1)
    return dataTrain, labelTrain, dataTest, labelTest

# def main():
#     (dataTrain, labelTrain), (dataTest, labelTest) = mnist.load_data()

#     # normalize datas
#     maxVal = numpy.max(dataTrain)
#     dataTrain = (dataTrain / maxVal).reshape(60000, 28, 28, 1)
#     dataTest = (dataTest / maxVal).reshape(10000, 28, 28, 1)

#     #building model
#     model = buildModel()

#     model.load_weights("logs/checkpoints/cp1.ckpt")

#     # cpCallBack = tensorflow.keras.callbacks.ModelCheckpoint(filepath="logs/checkpoints/cp1.ckpt", save_weights_only=True, verbose=1)

#     model.fit(dataTrain, labelTrain, batch_size=128, epochs = 10, validation_data=(dataTest, labelTest), callbacks=[cpCallBack])
#     model.summary()
#     model.evaluate(dataTest, labelTest)
#     return 0

def main():
    # dataTrain, labelTrain, dataTest, labelTest = dataPreprocessing()
    dataTest = load_images_from_folder(folder)
    dataTest=numpy.array(dataTest).reshape(4,28,28,1)
    labelTest=[0,1,2,3]
    labelTest=numpy.array(labelTest)
    hm = HomeworkMarking("logs/checkpoints/cp1.ckpt")
    print("\033[92m Evaluate the Model \033[0m")
    hm.evaluate(dataTest, labelTest)
    hm.predict(dataTest)
    # test = load_images_from_folder(folder)

    # print(test)
    # print(dataTest)
    # print(labelTrain)
    print(dataTest.shape, dataTest.dtype)
    print(labelTest.shape, labelTest.dtype)
    return 0

if __name__ == "__main__":
    main()