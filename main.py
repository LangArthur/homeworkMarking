#!/usr/bin/python3

import numpy
import tensorflow
from tensorflow.keras.datasets import mnist

from HomeworkMarking import *

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
    dataTrain, labelTrain, dataTest, labelTest = dataPreprocessing()

    hm = HomeworkMarking("logs/checkpoints/cp1.ckpt")
    print("\033[92m Evaluate the Model \033[0m")
    hm.evaluate(dataTest, labelTest)
    return 0

if __name__ == "__main__":
    main()