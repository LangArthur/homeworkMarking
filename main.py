#!/usr/bin/env python3

import numpy
import cv2
import os
import csv
import sys
from numpy import savetxt
from tensorflow.keras.datasets import mnist
from train_modelTest import dataPreprocessing
import tensorflow 


from src.HomeworkMarking import *

def displayHelp():
	print("Usage: ./main.py refFile") #TODO finish usage

def checkParameters(parameters):
	return len(parameters) > 1

def main():
    av = sys.argv
    if (checkParameters(av)):
        hw = HomeworkMarking(av[1])
        hw.run()
    else:
        displayHelp()
    return 0

if __name__ == "__main__":
    main()
