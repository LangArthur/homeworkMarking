#
# Created on Tue Mar 09 2021
#
# Arthur Lang
# ROIDetection.py
#

# https://nanonets.com/blog/handwritten-character-recognition/

# https://medium.com/@derrickfwang/printed-and-handwritten-text-extraction-from-images-using-tesseract-and-google-cloud-vision-api-ac059b62a535

import cv2
import numpy as np

class ROIDetection():

    def __init__(self):
        self._askwindow = True
        self._rois = []
        self.tmpRoi = [-1, -1, -1, -1]
        self.refImg = None

    def reset(self):
        self._rois = []
        self.tmpRoi = [-1, -1, -1, -1]
        self.refImg = None

    def crop(self, img):
        res = []
        for roi in self._rois:
            if (not(-1 in roi)):
                res.append(img[roi[1]:roi[3], roi[0]:roi[2]])
            else:
                print("Warning: Skip the following roi: " + str(roi))
        return res

    def getRoi(self, idx):
        if (idx < len(self._rois)):
            return self._rois[idx]
        else:
            return -1

    def clickCallback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.tmpRoi[0] = x
            self.tmpRoi[1] = y
        elif event == cv2.EVENT_LBUTTONUP:
            self.tmpRoi[2] = x
            self.tmpRoi[3] = y
            cv2.rectangle(self.refImg, (self.tmpRoi[2], self.tmpRoi[3]), (self.tmpRoi[0], self.tmpRoi[1]), (0,255,0), 1)
            self._rois.append(self.tmpRoi)
            self.tmpRoi = [-1, -1, -1, -1]

    def askRoi(self, ref):
        status = 0 # value to be return. 0 is to continue process, 1 is for stopping the problem
        self.refImg = ref.copy() # Copy to avoid square drawing on the picture
        cv2.namedWindow('Reference Image')
        cv2.setMouseCallback('Reference Image', self.clickCallback)
        while(self._askwindow):
            cv2.imshow('Reference Image', self.refImg)
            k = cv2.waitKey(20) & 0xFF
            if k == 27: # Escape
                self._askwindow = False
                status = 1
            elif k == 13: # Enter
                self._askwindow = False
        return status