#!/usr/bin/python3

import numpy
import tensorflow
from tensorflow.keras.datasets import mnist
import cv2
import os

from HomeworkMarking import *


def load_images_from_folder(folder):
	folder = os.getcwd() + folder
	images = []
	grays = []
	labels = []
	for filename in os.listdir(folder):
		img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
		# negative the picture and resize it to fit the training data
		img = cv2.resize(255-img, (28, 28))
		# cv2.imwrite("./minimize " + str(filename) + ".jpg", img)
		gray = (img/255.0)
		if gray is not None and img is not None:
			labels.append(float(os.path.splitext(filename)[0]))
			images.append(img)
			grays.append(gray)
	return images, grays, numpy.array(labels)


def dataPreprocessing():
	(dataTrain, labelTrain), (dataTest, labelTest) = mnist.load_data()
	# normalize datas
	maxVal = numpy.max(dataTrain)
	dataTrain = (dataTrain / maxVal).reshape(60000, 28, 28, 1)
	dataTest = (dataTest / maxVal).reshape(10000, 28, 28, 1)
	return dataTrain, labelTrain, dataTest, labelTest


def compareResult(predict, images):
	for pred, img in zip(predict, images):
		print(pred)
		cv2.imshow('output', img)
		cv2.waitKey(0)


def displayContour():
	# img = cv2.imread('test/testOperation.jpg', cv2.IMREAD_GRAYSCALE)
	# # cv2.imshow('read image', img)
	# # img = cv2.imread('test/childrenDigits/0.png')
	# # imgray = 255 - img
	# # imgray = cv2.cvtColor(imgray, cv2.COLOR_BGR2GRAY)
	# imgray = 255-img
	# # imgray = 255 - img
	# ret, thresh = cv2.threshold(imgray, 127, 255, 0)
	# im2,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# print(len(contours))
	# cv2.drawContours(imgray, contours, 0, (0, 255, 0), 3)
	# cv2.imshow('test', im2)

	image = cv2.imread('test/test.jpg')
	original_image = image
	img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	edged = cv2.Canny(img_gray, 30, 200)
	# cv2.waitKey(0)
	# cv2.imshow('Canny Edges After Contouring', edged)
	# cv2.waitKey(0)
	contours, hierarchy = cv2.findContours(edged.copy(),
	cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	for (i,c) in enumerate(contours):
		x,y,w,h= cv2.boundingRect(c)
		
		cropped_contour= original_image[y-5:y+h +5, x-5:x+w+5]
		image_name= "output_shape_number_" + str(i+1) + ".jpg"
		cv2.imwrite(image_name, cropped_contour)
		readimage= cv2.imread(image_name)
		cv2.imshow('Image', readimage)
		cv2.waitKey(0)
    
	cv2.destroyAllWindows()
	# for cnt in contours:
	# 	x, y, w, h = cv2.boundingRect(cnt)
	# 	# bound the images
	# 	cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
	# i = 0
	# for cnt in contours:
	# 	x, y, w, h = cv2.boundingRect(cnt)
	# 	# following if statement is to ignore the noises and save the images which are of normal size(character)
	# 	# In order to write more general code, than specifying the dimensions as 100,
	# 	# number of characters should be divided by word dimension
	# 	if w>20 and h>20:
	# 		# save individual images
	# 		cv2.imwrite(str(i)+".jpg",thresh1[y:y+h,x:x+w])
	# 		i=i+1
	# cv2.namedWindow('BindingBox', cv2.WINDOW_NORMAL)
	# cv2.imshow('BindingBox',image)
	# cv2.waitKey(0)
	# Start good
	# cv2.imshow('Canny Edges After Contouring', edged) 
	# cv2.waitKey(0) 
	# print("Number of Contours found = " + str(len(contours)))
	# cv2.drawContours(image, contours, -1, (0, 255, 0), 3) 
  
	# cv2.imshow('Contours', image) 
	# cv2.waitKey(0) 
	# cv2.destroyAllWindows()  
	# Good

	#     ret, im = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
	# contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# img = cv2.drawContours(res_img, contours, -1, (0,255,75), 2)
	# show_image(res_img)
	# cv2.waitKey(0)

def main():
	# folder = "/test/childrenDigits/"
	# dataTrain, labelTrain, dataTest, labelTest = dataPreprocessing()
	# sourcesImages, dataTest, labelTest = load_images_from_folder(folder)
	# dataTest=numpy.array(dataTest).reshape(4,28,28,1)

	# hm = HomeworkMarking("logs/checkpoints/cp1.ckpt")
	# hm.evaluate(dataTest, labelTest)
	# predict = hm.predict(dataTest)
	displayContour()

	# compareResult(predict, sourcesImages)

	# print(test)
	# print(dataTest)
	# print(labelTrain)
	# print(dataTest.shape, dataTest.dtype)
	# print(labelTest.shape, labelTest.dtype)
	return 0

if __name__ == "__main__":
	main()
