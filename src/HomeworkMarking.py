#
# Created on Tue Mar 09 2021
#
# Arthur Lang
# HomeworkMarking.py
#

import cv2

from src.ROIDetection import ROIDetection

class HomeworkMarking():

    def __init__(self, refDocPath, inputDir = "./input/"):
        self._refDocPath = refDocPath
        self._inputDir = inputDir
        self._roiDetector = ROIDetection()

    def __del__(self):
        cv2.destroyAllWindows()

    def setRefDoc(self, newDocPath):
        self._refDocPath = newDocPath

    def setInputDir(self, newInputDir):
        self._inputDir = newInputDir

    def run(self):
        refImgPath = cv2.imread(self._refDocPath)
        if (self._roiDetector.askRoi(refImgPath) == 0):
            crop = self._roiDetector.crop(refImgPath)
            for img in crop:
                cv2.imshow("ROI", img)
                cv2.waitKey(0)