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
from src.testModel import testModel

from tensorflow.keras.datasets import mnist
import tensorflow

class HomeworkMarking():

    def __init__(self, refDocPath, inputDir = "./input/"):
        self._refDocPath = refDocPath
        self._inputDir = inputDir
        self._roiDetector = ROIDetection()
        self._correction = []
        self._model = testModel()
        self._model = tensorflow.keras.models.load_model('D:\\JU\\homeworkMarking\\src\\modelTrainedOnChildrensdigit')
        self._outputColor = {
            0: (65, 158, 224),
            1: (0, 255, 0)
        }

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
        return numpy.reshape(res, (len(res), 28, 28, 1))

    def predict(self, pictures):
        data = self.preprocess(pictures)
        return self._model.predict_classes(data)

    def compare(self, current):
        score = []
        for i in range(len(current)):
            if (current[i] == self._correction[i]):
                score.append(1)
                # print("\033[92mCorrect!\033[0m")
            else:
                score.append(0)
                # print("\033[93mIncorrect :(\033[0m")
        return score
            # print("Your answer was: {} and the correct one was {}.".format(str(current[i]), str(self._correction[i])))

    def feedBack(self, img, scores):
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for i, score in enumerate(scores):
            roi = self._roiDetector.getRoi(i)
            cv2.rectangle(rgb, (roi[2], roi[3]), (roi[0], roi[1]), self._outputColor[score], 1)
            cv2.putText(rgb, str(self._correction[i]), (roi[0] + 2, roi[1] + 25), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, self._outputColor[score], 2, cv2.LINE_AA)
        cv2.imshow('Correction', rgb)
        cv2.waitKey(0)


    def correct(self):
        infiles = [f for f in listdir(self._inputDir) if isfile(join(self._inputDir, f))]
        for toCorrect in infiles:
            print("Correcting file " + toCorrect + "...")
            imgToCorrect = cv2.imread(self._inputDir + toCorrect, cv2.IMREAD_GRAYSCALE)
            crops = self._roiDetector.crop(imgToCorrect)
            # predict
            prediction = self.predict(crops)

            # compare with the input files
            # show correction
            score = self.compare(prediction)
            self.feedBack(imgToCorrect, score)
            # cv2.imshow("Correction", imgToCorrect)
            # cv2.waitKey(0)

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
