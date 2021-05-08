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

def checkArg(av):
    return len(av) == 2

def displayConfusionMatrix(matrix):
    print(matrix)
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
        print("Error: invalid number of arguments. Please specified a valid path of a model.", file=sys.stderr)
        return -1
    model = Model(modelPath = av[1])

    data, labels = load(preprocess=True)

    val = input("Run a cross validation ? [Y]es, [N]o\t")
    if (val == "Y"):
        result = crossValidation(numpy.array(data), numpy.array(labels), model)
        displayConfusionMatrix(numpy.sum(result, axis=0))
    elif (val == "N"):
        prediction = model.predict(data)
        displayConfusionMatrix(confusion_matrix(labels, prediction))

if __name__ == "__main__":
    evaluate()