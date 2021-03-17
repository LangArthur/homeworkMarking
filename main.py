#!/usr/bin/python3

import numpy
import tensorflow
# from tensorflow.keras.datasets import mnist
import cv2
import os
import csv 

from HomeworkMarking import *


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


def dataPreprocessing():
    (dataTrain, labelTrain), (dataTest, labelTest) = mnist.load_data()
    # normalize datas
    maxVal = numpy.max(dataTrain)
    dataTrain = (dataTrain / maxVal).reshape(60000, 28, 28, 1)
    dataTest = (dataTest / maxVal).reshape(10000, 28, 28, 1)
    return dataTrain, labelTrain, dataTest, labelTest

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



def main():
    # folder = "/test/childrenDigits/"
    # dataTrain, labelTrain, dataTest, labelTest = dataPreprocessing()
    # sourcesImages, dataTest, labelTest = load_images_from_folder(folder)
    # dataTest=numpy.array(dataTest).reshape(4,28,28,1)

    # hm = HomeworkMarking("logs/checkpoints/cp1.ckpt")
    # hm.evaluate(dataTest, labelTest)
    # predict = hm.predict(dataTest)
    # displayContour()
    image_test,labels_test,images_train,labels_train=load_dataset('Double digits resized')
    with open('Dataset_csv/train_data', 'w') as train_dataset:
        write = csv.writer(train_dataset) 
        write.writerow(images_train) 
    with open('Dataset_csv/train_label', 'w') as train_label:
        write = csv.writer(train_label) 
        write.writerow(labels_train) 
    with open('Dataset_csv/test_label', 'w') as test_label:
        write = csv.writer(test_label) 
        write.writerow(labels_test) 
    with open('Dataset_csv/test_data', 'w') as test_dataset:
        write = csv.writer(test_dataset) 
        write.writerow(image_test) 

    # compareResult(predict, sourcesImages)

    # print(test)
    # print(dataTest)
    # print(labelTrain)
    # print(dataTest.shape, dataTest.dtype)
    # print(labelTest.shape, labelTest.dtype)
    return 0

if __name__ == "__main__":
    main()
