#!/usr/bin/env python3

import numpy
import cv2
import os
import csv
import sys

from src.HomeworkMarking import *


def resize(directory):
    # folder = os.getcwd() + folder
    images = []
    grays = []
    labels = []
    for folder in os.listdir(directory):
        # print(os.path.join(directory,folder))
        path= 'D:\\JU\\ML Project\\Double digits resized'
        path = os.path.join(path,folder)
        if not os.path.exists(path):
            os.mkdir(path)
        for number in os.listdir(os.path.join(directory,folder)):
            i=0
            number_path = os.path.join(path, number)
            to_read_from = os.path.join(directory,folder, number)
            print(number_path)
            for image in os.listdir(to_read_from):
                # i=0
                img = cv2.imread(os.path.join(directory,folder, number, image))
                img = cv2.resize(img, (28,28))
                # cv2.imshow("pic",img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # print(number_path)
                
                write_path = number_path+'\\'+str(i)+'.jpg'
                i+=1
                # print(" write path:",write_path)
                cv2.imwrite(write_path, img)
                i+=1
    # 	img = cv2.imread(number_path)
    # # 	# negative the picture and resize it to fit the training data
    # 	img = cv2.resize( img,(28, 28))
    # 	cv2.imshow(img)
    # 	# cv2.imwrite("./minimize " + str(filename) + ".jpg", img)
    # 	gray = (img/255.0)
    # 	if gray is not None and img is not None:
    # 		labels.append(float(os.path.splitext(filename)[0]))
    # 		images.append(img)
    # 		grays.append(gray)
    # return images, grays, numpy.array(labels)

def load_dataset(folder):
    images_train=[]
    labels_train=[]
    image_test=[]
    labels_test=[]
    for item in os.listdir(folder):
        number_path = os.path.join(folder,item)
        # print(number_path)
        for number in os.listdir(number_path):
            images_path =os.path.join(number_path, number)
            for image in os.listdir(images_path):
                img = cv2.imread(os.path.join(images_path,image))
                if item == "train" and img is not None:
                    images_train.append(img)
                    labels_train.append(number)
                elif item == "test" and img is not None:
                    image_test.append(img)
                    labels_test.append(number)
    return image_test,labels_test,images_train,labels_train


def compareResult(predict, images):
    for pred, img in zip(predict, images):
        print(pred)
        cv2.imshow('output', img)
        cv2.waitKey(0)

def displayHelp():
    print("Usage: ./main.py refFile")
    print("Correct a set of assignment following a correction.")
    print("\nrefFile\t\tpath to the correction file.")
    print("\nBy default, the programm will look for assignments to correct in the folder \"input/\".")

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
