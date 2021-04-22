#!/usr/bin/env python3
#
# Created on Mon Apr 12 2021
#
# Arthur Lang
# evaluate.py
#

import sys
import numpy
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.Model import Model
from src.dataset import load

"""
#TODO think about how will be the evaluation method:

- generate on test dataset (with children data), test with different training dataset.
- generate one dataset for training and testing, but change the model we load.

is it enough to return confusion matrix on a cross validation ?

"""

def checkArg(av):
    return len(av) == 2

def displayConfusionMatrix(matrix):
    seaborn.set(font_scale=1.4) # for label size
    seaborn.heatmap(matrix, annot=True, annot_kws={"size": 16}) # font size
    plt.show()

def crossValidation(data, target, model, split_size=5):
    results = []
    kf = KFold(n_splits=split_size)
    for trainIdx, valIdx in kf.split(data, target):
        trainData = data[trainIdx]
        trainTarget = target[trainIdx]
        testData = data[valIdx]
        testTarget = target[valIdx]

        model.trainWithoutValidation(trainData, trainTarget)
        predict = model.predict(testData)
        results.append(confusion_matrix(testTarget, predict))
    return results

def evaluate():
    av = sys.argv
    if (not(checkArg(av))):
        print("Error: invalid number of arguments.", file=sys.stderr)
        return -1
    model = Model(modelPath = av[1])

    data, labels = load(preprocess=True)

    result = crossValidation(numpy.array(data), numpy.array(labels), model)
    # print()
    displayConfusionMatrix(numpy.sum(result, axis=0))
    # return 0

if __name__ == "__main__":
    evaluate()