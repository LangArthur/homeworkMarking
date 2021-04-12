#!/usr/bin/env python3
#
# Created on Mon Apr 12 2021
#
# Arthur Lang
# evaluate.py
#

import sys
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from src.Model import Model

"""
#TODO think about how will be the evaluation method:

- generate on test dataset (with children data), test with different training dataset.
- generate one dataset for training and testing, but change the model we load.

is it enough to return confusion matrix on a cross validation ?

"""

def checkArg(av):
    return len(av) == 1

def evaluate():
    av = sys.argv
    if (not(checkArg(av))):
        return -1
    model = Model(modelPath = av[1])

    #TODO generate dataset
    result = crossValidation(data, target, model)
    print ("Cross-validation result: %s" % result)
    return 0

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

if __name__ == "__main__":
    evaluate()