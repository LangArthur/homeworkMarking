#
# Created on Wed Mar 17 2021
#
# Arthur Lang
# Model.py
#

from enum import Enum
import tensorflow
import numpy
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization

## Model
# class encapsulating a kears model
class Model():

    ## constructor
    # @param weightPath: path where the weights are saved. It will load them automatically
    # @param model: filepath to load a complet model with associated weights (used for retraining for example).
    def __init__(self, weightPath = None, modelPath = None):
        self._hasWeight = False
        if (modelPath != None):
            self._model = tensorflow.keras.models.load_model(modelPath)
            self._hasWeight = True
        else:
            self._model = self._buildModel()
            if (weightPath != None):
                self._model.load_weights(weightPath)
                self._hasWeight = True
        self._outputWeightPath = "weights/lastTraining/cp1.ckpt"

    ## setOutputWeightPath
    # setter for outpout weight directory.
    def setOutputWeightPath(self, path):
        self._outputWeightPath = path

    ## _buildModel
    # build and return the model
    def _buildModel(self):
        model = Sequential()
        model.add(Conv2D(32,(3,3), strides=(1, 1),  activation="relu",input_shape = (28,28,1),data_format = "channels_last", use_bias = True))
        model.add(Conv2D(32,(3,3), strides=(1, 1),  activation="relu", use_bias = True))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128,(3,3), strides=(1, 1),  activation="relu", use_bias = True))
        model.add(Conv2D(128,(3,3), strides=(1, 1),  activation="relu", use_bias = True))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(256,activation = "relu", use_bias = True))
        model.add(Dropout(0.5))
        model.add(Dense(11,activation = "softmax",use_bias = True))

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    ## train
    # train the model and save the gotten weights
    def train(self, data, labels, dataTest, labelTest):
        cpCallBack = tensorflow.keras.callbacks.ModelCheckpoint(filepath=self._outputWeightPath, save_weights_only=True, verbose=1)
        self._model.fit(data, labels, batch_size=128, epochs = 10, validation_data=(dataTest, labelTest), callbacks=[cpCallBack])

    ## trainWithoutValidation
    # train the model without saving weights and without validation_data
    def trainWithoutValidation(self, data, labels):
        self._model.fit(data, labels, batch_size=128, epochs = 10)

    ## fit
    # fit the model without saving the weights
    def fit(self, data, labels, dataTest, labelTest):
        self._model.fit(data, labels, batch_size=128, epochs = 10, validation_data=(dataTest, labelTest))

    ## evaluate
    # evaluate the model
    def evaluate(self, testData, testLabel):
        # self._model.summary()
        return self._model.evaluate(testData, testLabel)

    ## predict
    # to a prediction on a set of data
    # @param testData: the data you want to make prediction on
    # @return an array with all the predicted classes
    def predict(self, testData):
        return numpy.argmax(self._model.predict(testData), axis=-1)
        # return self._model.predict_classes(testData)
    
    ## save
    # save the model in a file
    # @param path: path where to save the model
    def save(self, path):
        self._model.save(path)