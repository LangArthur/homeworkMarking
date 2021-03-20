#
# Created on Tue Mar 09 2021
#
# Arthur Lang
# ROIDetection.py
#

import cv2
import numpy as np

## ROIDetection
# object managing all ROIs
class ROIDetection():

    ## constructor
    def __init__(self):
        self._askwindow = True # boolean for runing ask window
        self._rois = []
        self._tmpRoi = [-1, -1, -1, -1]
        self._refImg = None # image on wich ROIs are based on.

    ## reset
    # reset the ROIDetection as if it was freshly build
    def reset(self):
        self._rois = []
        self._tmpRoi = [-1, -1, -1, -1]
        self._refImg = None

    ## crop
    # crop ROI of an image
    # @param img: image to crop
    # @return crop(s) from an images, follwing the ROIs
    def crop(self, img):
        res = []
        for roi in self._rois:
            if (not(-1 in roi)):
                res.append(img[roi[1]:roi[3], roi[0]:roi[2]])
            else:
                print("Warning: Skip the following roi: " + str(roi))
        return res

    ## getROi
    # getter for a specific roi
    # @param idx: index of the roi
    # @return the roi or -1 if idx do not correspond to a valid index.
    def getRoi(self, idx):
        if (idx < len(self._rois)):
            return self._rois[idx]
        else:
            return -1

    ## _clickCallback
    # function call when user click on a picture    
    def _clickCallback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._tmpRoi[0] = x
            self._tmpRoi[1] = y
        elif event == cv2.EVENT_LBUTTONUP:
            self._tmpRoi[2] = x
            self._tmpRoi[3] = y
            cv2.rectangle(self._refImg, (self._tmpRoi[2], self._tmpRoi[3]), (self._tmpRoi[0], self._tmpRoi[1]), (0,255,0), 1)
            self._rois.append(self._tmpRoi)
            self._tmpRoi = [-1, -1, -1, -1]

    ## askRoi
    # open the view to select rois
    # @param ref: picture of reference
    # @return status of the view. 1 represente something wrong, so asking to stop the program.
    def askRoi(self, ref):
        status = 0 # value to be return. 0 is to continue process, 1 is for stopping the problem
        self._refImg = ref.copy() # Copy to avoid square drawing on the picture
        cv2.namedWindow('Reference Image')
        cv2.setMouseCallback('Reference Image', self._clickCallback)
        while(self._askwindow):
            cv2.imshow('Reference Image', self._refImg)
            k = cv2.waitKey(20) & 0xFF
            if k == 27: # Escape
                self._askwindow = False
                status = 1
            elif k == 13: # Enter
                self._askwindow = False
        return status