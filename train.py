#!/usr/bin/env python3
#
# Created on Wed Mar 17 2021
#
# Arthur Lang
# train.py
#

import numpy
from tensorflow.keras.datasets import mnist

from src.Model import Model

## load_dataset
# load custom dataset (mnist + children)
def load_dataset(folder):
    images_train=[]
    labels_train=[]
    images_test=[]
    labels_test=[]
    for item in os.listdir(folder):
        number_path = os.path.join(folder,item)
        print(number_path)
        for number in os.listdir(number_path):
            images_path =os.path.join(number_path, number)
            print(images_path)
            for image in os.listdir(images_path):
                img = cv2.imread(os.path.join(images_path,image), cv2.IMREAD_GRAYSCALE)
                # img = 255-img
                if item == "train" and img is not None:
                    images_train.append(img)
                    labels_train.append(number)
                elif item == "test" and img is not None:
                    images_test.append(img)
                    labels_test.append(number)
    
    return  images_train, labels_train,images_test,labels_test

def set_shape(data):
    res = []
    for item in data:
        item=item/255
        item=item.reshape(28,28,1)
        res.append(item)
    res=numpy.array(res)
    res=res.reshape(len(res),28,28,1)
    return res

def to_float(data):
    res=[]
    for item in data:
        res.append(float(item))
    res=numpy.array(res)
    return res


def dataPreprocessing():
    # Mnist dataset
    (dataTrain, labelTrain), (dataTest, labelTest) = mnist.load_data()
    # normalize datas
    maxVal = numpy.max(dataTrain)
    dataTrain = (dataTrain / maxVal).reshape(60000, 28, 28, 1)
    dataTest = (dataTest / maxVal).reshape(10000, 28, 28, 1)
    return dataTrain, labelTrain, dataTest, labelTest

## train
# do a training session, saving the weights
def train():
    dataTrain, labelTrain, dataTest, labelTest = dataPreprocessing()

    nt = Model()
    nt.setOutputWeightPath("ressources/weights/retraining-children/")
    nt.train(dataTrain, labelTrain, dataTest, labelTest)
    return 0

if __name__ == "__main__":
    train()