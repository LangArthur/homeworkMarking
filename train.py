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
    nt.setOutputWeightPath("logs/simple-mnist/")
    nt.train(dataTrain, labelTrain, dataTest, labelTest)
    return 0

if __name__ == "__main__":
    train()