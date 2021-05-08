#
# Created on Thu Apr 22 2021
#
# Arthur Lang
# dataset.py
#

import os
import PIL
import cv2
import numpy
from numpy import expand_dims
from matplotlib import pyplot
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator  as igen


## load_dataset
# load custom dataset (mnist + children)
def load_dataset(folder):
    data = []
    label = []
    images_train = []
    labels_train = []
    images_test = []
    labels_test = []
    # iterate through each folders
    for digitFolder in os.listdir(folder):
        if (digitFolder != "unlabeled"):
            images_path = os.path.join(folder, digitFolder)
            # iterate throught images
            for image in os.listdir(images_path):
                img = cv2.imread(os.path.join(images_path,image), cv2.IMREAD_GRAYSCALE)
                data.append(img)
                label.append(int(digitFolder))
                # # push to train tdataset
                # if item == "train" and img is not None:
                #     images_train.append(img)
                #     labels_train.append(int(number))
                # # push to test tdataset
                # elif item == "test" and img is not None:
                #     images_test.append(img)
                #     labels_test.append(int(number))
    # return  images_train, labels_train, images_test, labels_test
    return data, label
def data_augmenation(currentPath, augmentedImagesPath):

    resultsPath=os.path.join(augmentedImagesPath, "AugmentedDataset")
    print("results path:",resultsPath)
    if(not os.path.exists(resultsPath)):
        os.mkdir(resultsPath)
    
    for folder in os.listdir(currentPath):
        pathToSave=os.path.join(resultsPath,folder) 
        if(not os.path.exists(pathToSave)):
            os.mkdir(pathToSave)
        i=0
        if not (len(folder)>2):
            imagesPath=os.path.join(currentPath,folder)
            # print(imagesPath)
            for image in os.listdir(imagesPath):
                imagePath=os.path.join(imagesPath, image)
                img = load_img(imagePath, color_mode="grayscale")
                # convert to numpy array
                data = img_to_array(img)
                print(data.shape)
                # expand dimension to one sample
                samples = expand_dims(data, 0)
                print(samples.shape)
                # img = cv2.imread(os.path.join(imagesPath,image), cv2.IMREAD_GRAYSCALE)
                # print(type(img))
                # img.astype('float32')
                shift=0.2
                datagen=igen(width_shift_range=shift, height_shift_range=shift,zca_whitening=True)
                fit = datagen.fit(samples)
                print("path to save:",pathToSave)
                for batch in datagen.flow(samples, batch_size = 1, 
                      save_to_dir =pathToSave,  
                      save_prefix ='digit{}'.format(i), save_format ='jpeg'):
                            i += 1
                            print(i)
                            if i > 5: 
                                i=0
                                break
                # image = np.squeeze(image)
                # pyplot.imshow(fit.astype('uint8'))
                # break
                # datagen=igen(zca_whitening=True)
                # img=datagen.fit(img)
                # zca_whitening=True
                # cv2.imwrite(os.path.join(resultsImagesPath , 'augmented{}.jpg'.format(i)), img)
                

    #    print(folder)
    #    if isinstance(5, int)
    #    images_path = os.path.join(folder, digitFolder)
    #    resultsImagesPath=os.path.join(rasultsPath,digitFolder)
    #    for image in os.listdir(images_path):
    #        img = cv2.imread(os.path.join(images_path,image), cv2.IMREAD_GRAYSCALE)
    #        img.astype('float32')
    #        shift=0.2
    #        datagen=igen(width_shift_range=shift, height_shift_range=shift)
    #        fit = datagen.fit(img)
    #        datagen=igen(zca_whitening=True)
    #        img=datagen.fit(img)
    #        zca_whitening=True
    #        cv2.imwrite(os.path.join(resultsImagesPath , 'waka{}.jpg'.format(i)), img)


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

# data_augmenation("C:\\Users\\Madalina Aldea\\Desktop\\homeworkMarking\\resources\\datasets\\Children_dataset\\", "C:\\Users\\Madalina Aldea\\Desktop\\homeworkMarking\\resources\\datasets\\")
# print(os.path.isdir("C:\\Users\\Madalina Aldea\\Desktop\\homeworkMarking\\resources\\datasets\\Children_dataset"))
# os.mkdir("C:\\Users\\Madalina Aldea\\Desktop\\homeworkMarking\\resources\\datasets\\Children_dataset\\test")