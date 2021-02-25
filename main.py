#!/usr/bin/python3

import numpy
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.datasets import mnist

def main():
    (dataTrain, labelTrain), (dataTest, labelTest) = mnist.load_data()

    # normalize datas
    maxVal = numpy.max(dataTrain)
    dataTrain = (dataTrain / maxVal).reshape(60000, 28, 28, 1)
    dataTest = (dataTest / maxVal).reshape(10000, 28, 28, 1)

    #building model
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size=(3,3), activation='relu'))
    # model.add(Conv2D(filters = 32, kernel_size=(3,3), activation='relu', input_shape=dataTrain[0].shape()))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(dataTrain, labelTrain, batch_size=128, epochs = 10, validation_data=(dataTest, labelTest))
    model.summary()
    model.evaluate(dataTest, labelTest)
    return 0

if __name__ == "__main__":
    main()