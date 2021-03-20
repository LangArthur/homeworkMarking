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
    
    def save(self, path):
        self._model.save(path)