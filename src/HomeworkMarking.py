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

## HomeworkMarking
# main class of the Homework marking project
class HomeworkMarking():

    ## constructor
    # @param refDocPath: path for the correction image.
    # @param inputDir: directory in wich all the pictures for correction are. By default on the input directory.
    def __init__(self, refDocPath, inputDir = "./input/"):
        self._refDocPath = refDocPath # path to the correction document
        self._inputDir = inputDir
        self._roiDetector = ROIDetection()
        self._correction = []
        self._model = Model("logs/checkpoints/cp1.ckpt")
        self._outputColor = {
            0: (65, 158, 224),
            1: (0, 255, 0)
        }
        # 0 is the color when answer is wrong, 1 when it's good

    def __del__(self):
        cv2.destroyAllWindows()

    ## setRefDoc
    # setter for the path of the correction.
    def setRefDoc(self, newDocPath):
        self._refDocPath = newDocPath

    ## setInputDir
    # setter for the input directory.
    def setInputDir(self, newInputDir):
        self._inputDir = newInputDir

    ## _preprocess
    # preprocess the data to fit the model
    # @param data: data to process
    # @return numpy array of the preprocess data
    def _preprocess(self, data):
        res = []
        for img in data:
            resize = 255 - (cv2.resize(img, (28, 28)))
            res.append(cv2.threshold(resize, 120, 255, cv2.THRESH_BINARY)[1] / 255)
        return numpy.reshape(res, (len(res), 28, 28, 1))

    ## _predict
    # predict class for a set of images
    # @param pictures: array containing the pictures to predict
    # @retur the prediction of the model
    def _predict(self, pictures):
        data = self._preprocess(pictures)
        return self._model.predict(data)

    ## _compare
    # compare the correction with the values get from 
    # @param current
    # @return: an array containing the result. 1 stand for a good answer, 0 for a bad one. The digits are sorted depending of the ROI order.
    def _compare(self, current):
        score = []
        for i in range(len(current)):
            if (current[i] == self._correction[i]):
                score.append(1)
            else:
                score.append(0)
        # this print is for checking the algorithm detection
        print("Your answer was: {} and the correct one was {}.".format(str(current[i]), str(self._correction[i])))
        return score

    ## _feedBack
    # display feedback on the original picture
    #
    def _feedBack(self, img, scores):
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for i, score in enumerate(scores):
            roi = self._roiDetector.getRoi(i)
            cv2.rectangle(rgb, (roi[2], roi[3]), (roi[0], roi[1]), self._outputColor[score], 1)
            cv2.putText(rgb, str(self._correction[i]), (roi[0] + 2, roi[1] + 25), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, self._outputColor[score], 2, cv2.LINE_AA)
        cv2.imshow('Correction', rgb)
        cv2.waitKey(0)

    ## _correct
    # read throught the input folder all the pictures and correct them
    def _correct(self):
        infiles = [f for f in listdir(self._inputDir) if isfile(join(self._inputDir, f))]
        for toCorrect in infiles:
            print("\033[94mCorrecting file " + toCorrect + "...\033[0m")
            imgToCorrect = cv2.imread(self._inputDir + toCorrect, cv2.IMREAD_GRAYSCALE)
            crops = self._roiDetector.crop(imgToCorrect)
            # predict
            prediction = self._predict(crops)
            # compare with the input files
            score = self._compare(prediction)
            # show correction
            self._feedBack(imgToCorrect, score)

    ## run
    # run a correction session.
    def run(self):
        refImg = cv2.imread(self._refDocPath, cv2.IMREAD_GRAYSCALE)
        if (self._roiDetector.askRoi(refImg) == 0):
            crops = self._roiDetector.crop(refImg)
            self._correction = self._predict(crops)
            self._correct()