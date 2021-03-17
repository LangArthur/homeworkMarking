#
# Created on Tue Mar 09 2021
#
# Arthur Lang
# HomeworkMarking.py
#

import cv2
import numpy

from src.ROIDetection import ROIDetection
from src.Model import Model

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

    def setCorrection(self, crops):
        data = []
        # preprocessing
        for crop in crops:
            resize = 255 - (cv2.resize(crop, (28, 28)))
            data.append(cv2.threshold(resize, 127, 255, cv2.THRESH_BINARY)[1])

        # TODO Need to normalize the data here

        # for p in data:
        #     cv2.imshow("test", p)
        #     print(numpy.shape(p))
        #     cv2.waitKey(0)
        # data = [numpy.asarray(x).reshape(1, 28, 28, 1) for x in data]
        # print(data[0][0][0])
        # print(self._model.predict(data))

    def run(self):
        refImgPath = cv2.imread(self._refDocPath, cv2.IMREAD_GRAYSCALE)
        if (self._roiDetector.askRoi(refImgPath) == 0):
            crops = self._roiDetector.crop(refImgPath)
            self.setCorrection(crops)
            # for img in crop:
            #     cv2.imshow("ROI", img)
            #     cv2.waitKey(0)