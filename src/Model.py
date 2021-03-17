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

# class HMMode(Enum):
#     TRAINING = 0
#     RUNNING = 1

class Model():

    def __init__(self, weightPath = None):
        # self._mode = HMMode.TRAINING
        self._model = self.buildModel()
        if (weightPath != None):
            self._model.load_weights(weightPath)
            self._hasWeight = True
        else:
            self._hasWeight = False
        # self._action = {
        #     HMMode.TRAINING: self._train,
        #     HMMode.RUNNING: self._run
        # }
        self.outputWeightPath = "logs/checkpoints/cp1.ckpt"

    def setOutputWeightPath(self, path):
        self.outputWeightPath = path

    def setMode(self, newMode):
        self.mode = newMode

    def buildModel(self):
        model = Sequential([
            Conv2D(filters = 32, kernel_size=(3,3), activation='relu'),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPool2D(2, 2),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(10, activation='softmax'),
        ])

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, data, labels, dataTest, labelTest):
        cpCallBack = tensorflow.keras.callbacks.ModelCheckpoint(filepath=self.outputWeightPath, save_weights_only=True, verbose=1)
        self._model.fit(data, labels, batch_size=128, epochs = 10, validation_data=(dataTest, labelTest), callbacks=[cpCallBack])

    def run(self, data, labels, dataTest, labelTest):
        self._model.fit(data, labels, batch_size=128, epochs = 10, validation_data=(dataTest, labelTest))

    def evaluate(self, testData, testLabel):
        # self._model.summary()
        self._model.evaluate(testData, testLabel)

    def predict(self, testData):
        return self._model.predict_classes(testData)