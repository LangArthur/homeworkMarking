#
# Created on Wed Mar 17 2021
#
# Arthur Lang
# Model.py
#

from enum import Enum
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense

## Model
# class encapsulating a kears model
class Model():

    def __init__(self, weightPath = None):
        self._model = self._buildModel()
        if (weightPath != None):
            self._model.load_weights(weightPath)
            self._hasWeight = True
        else:
            self._hasWeight = False
        self._outputWeightPath = "logs/checkpoints/cp1.ckpt"

    def setOutputWeightPath(self, path):
        self._outputWeightPath = path

    def _buildModel(self):
        model = Sequential([
            Conv2D(filters = 32, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPool2D(2, 2),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(100, activation='softmax'),
        ])

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, data, labels, dataTest, labelTest):
        cpCallBack = tensorflow.keras.callbacks.ModelCheckpoint(filepath=self._outputWeightPath, save_weights_only=True, verbose=1)
        self._model.fit(data, labels, batch_size=128, epochs = 10, validation_data=(dataTest, labelTest), callbacks=[cpCallBack])

    def fit(self, data, labels, dataTest, labelTest):
        self._model.fit(data, labels, batch_size=128, epochs = 10, validation_data=(dataTest, labelTest))

    def evaluate(self, testData, testLabel):
        # self._model.summary()
        self._model.evaluate(testData, testLabel)

    def predict(self, testData):
        return self._model.predict_classes(testData)