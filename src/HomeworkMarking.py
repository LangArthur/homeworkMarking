#
# Created on Tue Mar 09 2021
#
# Arthur Lang
# HomeworkMarking.py
#

from os import listdir
from os.path import isfile, join
import cv2
import numpy

from src.ROIDetection import ROIDetection
from src.Model import Model

from tensorflow.keras.datasets import mnist

class HomeworkMarking():

    def __init__(self, refDocPath, inputDir = "./input/"):
        self._refDocPath = refDocPath
        self._inputDir = inputDir
        self._roiDetector = ROIDetection()
        self._correction = []
        self._model = Model("logs/checkpoints/cp1.ckpt")

    def __del__(self):
        cv2.destroyAllWindows()

    def setRefDoc(self, newDocPath):
        self._refDocPath = newDocPath

    def setInputDir(self, newInputDir):
        self._inputDir = newInputDir

    def preprocess(self, data):
        res = []
        for img in data:
            resize = 255 - (cv2.resize(img, (28, 28)))
            res.append(cv2.threshold(resize, 120, 255, cv2.THRESH_BINARY)[1] / 255)
        return res

    def predict(self, pictures):
        return [3, 7, 9, 9]
        # data = self.preprocess(pictures)

        # data = 255 - numpy.reshape(crops, (len(crops), 28, 28, 1))
        # _, data = cv2.threshold(data, 127, 255, cv2.THRESH_BINARY) / 255
        # numpy.reshape(data, (len(data), 28, 28, 1))
        # data = [numpy.asarray(x).reshape(1, 28, 28, 1) for x in data]
        # for p in data:
        #     cv2.imshow("test", p)
        #     print(numpy.shape(p))
        #     cv2.waitKey(0)

        # return self._model.predict(data)

    def compare(self, current):
        for i in range(len(current)):
            if (current[i] == self._correction[i]):
                print("Correct!")
            else:
                print("Incorrect :(")

    def correct(self):
        infiles = [f for f in listdir(self._inputDir) if isfile(join(self._inputDir, f))]
        for toCorrect in infiles:
            print("Correcting file " + toCorrect + "...")
            imgToCorrect = cv2.imread(self._inputDir + toCorrect, cv2.IMREAD_GRAYSCALE)
            crops = self._roiDetector.crop(imgToCorrect)
            for crop in crops:
                cv2.imshow(toCorrect, crop)
                cv2.waitKey(0)
                cv2.destroyWindow(toCorrect)
            # predict
            prediction = self.predict(crops)
            self.compare(prediction)

    def run(self):
        refImg = cv2.imread(self._refDocPath, cv2.IMREAD_GRAYSCALE)
        if (self._roiDetector.askRoi(refImg) == 0):
            crops = self._roiDetector.crop(refImg)
            self._correction = self.predict(crops)
            self.correct()

def dataPreprocessing():
    # Mnist dataset
    (dataTrain, labelTrain), (dataTest, labelTest) = mnist.load_data()
    # normalize datas
    maxVal = numpy.max(dataTrain)
    dataTrain = (dataTrain / maxVal).reshape(60000, 28, 28, 1)
    dataTest = (dataTest / maxVal).reshape(10000, 28, 28, 1)
    return dataTrain, labelTrain, dataTest, labelTest
