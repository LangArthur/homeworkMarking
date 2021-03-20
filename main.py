#!/usr/bin/env python3

import numpy
import cv2
import os
import csv
import sys
from numpy import savetxt

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
    images_test=[]
    labels_test=[]
    for item in os.listdir(folder):
        number_path = os.path.join(folder,item)
        print(number_path)
        for number in os.listdir(number_path):
            images_path =os.path.join(number_path, number)
            print(images_path)
            for image in os.listdir(images_path):
                img = cv2.imread(os.path.join(images_path,image), cv2.IMREAD_GRAYSCALE)
                if item == "train" and img is not None:
                    images_train.append(img)
                    labels_train.append(number)
                elif item == "test" and img is not None:
                    images_test.append(img)
                    labels_test.append(number)
    # return numpy.reshape(images_train, (len(images_train), 28, 28, 1)),numpy.reshape(labels_train, (len(labels_train), 28, 28, 1)),numpy.reshape(images_test, (len(images_test), 28, 28, 1)),numpy.reshape(labels_test, (len(labels_test), 28, 28, 1))
    
    return  images_train, labels_train,images_test,labels_test
    

def compareResult(predict, images):
    for pred, img in zip(predict, images):
        print(pred)
        cv2.imshow('output', img)
        cv2.waitKey(0)

## displayHelp
# print help
def displayHelp():
    print("Usage: ./main.py refFile")
    print("Correct a set of assignment following a correction.")
    print("\nrefFile\t\tpath to the correction file.")
    print("\nBy default, the programm will look for assignments to correct in the folder \"input/\".")

## CheckParameters
# check if the paramters are good or not
# @return true or false
def checkParameters(parameters):
	return len(parameters) > 1

def set_shape(data):
    res = []
    for item in data:
        item=item/255
        # print(item)
        item=item.reshape(28,28,1)
        res.append(item)
    res=numpy.array(res)
    res=res.reshape(len(res),28,28,1)
    return res

def to_float(data):
    res=[]
    for item in data:
        res.append(float(item))
    res=numpy.array(res)
    return res


def main():
    av = sys.argv
    if (checkParameters(av)):
        hw = HomeworkMarking(av[1])
        hw.run()
    else:
        displayHelp()
    return 0
    # path = 'D:\\JU\\ML Project\\homeworkMarking\\Double digits resized'
    # images_train,labels_train,images_test,labels_test = load_dataset(path)

    # for i in range(len(images_test)):
    #     for j in range(len(images_test[i])):
    #         if isinstance(images_test[i][j], str):
    #             print(images_test[i][j])

    # images_train = set_shape(images_train)
    # print(images_train.shape)
    # images_test= set_shape(images_test)
    # print(images_test.shape)
    # labels_train=numpy.array(labels_train)
    # labels_test=numpy.array(labels_test)

    # labels_train=to_float(labels_train)
    # labels_test=to_float(labels_test)

    # for item in labels_train:
    #     if isinstance(item, str):
    #         print("yes")
    # print(len(labels_train))
    # print((labels_test))
    # print(images_test.shape, images_train.shape,labels_test.shape,labels_train.shape)

    # model = Model()
    # model.buildModel()
    # model.train(images_train,labels_train,images_test,labels_test)
    # print(labels_test[-1])
    # print(labels_train[-1])
    
# def main():
#     # folder = "/test/childrenDigits/"
#     # dataTrain, labelTrain, dataTest, labelTest = dataPreprocessing()
#     # sourcesImages, dataTest, labelTest = load_images_from_folder(folder)
#     # dataTest=numpy.array(dataTest).reshape(4,28,28,1)

#     # hm = HomeworkMarking("logs/checkpoints/cp1.ckpt")
#     # hm.evaluate(dataTest, labelTest)
#     # predict = hm.predict(dataTest)
#     # displayContour()
#     image_test,labels_test,images_train,labels_train=load_dataset('Double digits resized')
#     with open('Dataset_csv/train_data', 'w') as train_dataset:
#         write = csv.writer(train_dataset) 
#         write.writerow(images_train) 
#     with open('Dataset_csv/train_label', 'w') as train_label:
#         write = csv.writer(train_label) 
#         write.writerow(labels_train) 
#     with open('Dataset_csv/test_label', 'w') as test_label:
#         write = csv.writer(test_label) 
#         write.writerow(labels_test) 
#     with open('Dataset_csv/test_data', 'w') as test_dataset:
#         write = csv.writer(test_dataset) 
#         write.writerow(image_test) 

#     # compareResult(predict, sourcesImages)

#     # print(test)
#     # print(dataTest)
#     # print(labelTrain)
#     # print(dataTest.shape, dataTest.dtype)
#     # print(labelTest.shape, labelTest.dtype)
#     return 0

if __name__ == "__main__":
    main()
