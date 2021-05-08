#
# Created on Thu Apr 22 2021
#
# Arthur Lang
# dataset.py
#

import os
import cv2
import numpy

## load_dataset
# load custom dataset (mnist + children)
def load_dataset(folder):
    data = []
    label = []
    # iterate through each folders
    for digitFolder in os.listdir(folder):
        if (digitFolder != "unlabeled" and digitFolder != "10"):
            images_path = os.path.join(folder, digitFolder)
            # iterate throught images
            for image in os.listdir(images_path):
                img = cv2.imread(os.path.join(images_path, image), cv2.IMREAD_GRAYSCALE)
                data.append(img)
                label.append(int(digitFolder))
    return data, label

## set_shape
# preprocess pictures from children to make them fit with mnist dataset
def set_shape(data):
    res = []
    for item in data:
        item = item / 255
        item = item.reshape(28, 28, 1)
        res.append(item)
    res = numpy.array(res).reshape(len(res), 28, 28, 1)
    return res

def unionShuffle(a, b):
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(a)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(b)

## load
# load and return the children dataset
# @param preprocess: allow preprocessing to fit the dataset with mnist dataset
def load(preprocess = False):
    if (preprocess):
        data, labels = load_dataset("resources/datasets/Children_dataset")
        unionShuffle(data, labels)
        return  set_shape(data), labels
    else:
        return load_dataset("resources/datasets/Children_dataset")