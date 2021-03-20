# HomeworkMarking
A tool to help teachers to correct homeworks.

Note that for the moment it only work with digits.

## Installation
We recommande to use pip for installing all the requirements

```
pip install -r requirement.txt
```

## Usage

```
./main.py refFile
```

The refFile is the picture of the assignment filled by the teacher: it contains the good answers.

The first screen will be the picture of this file. First, you need to select (with your mouse) the Region Of Intereset (ROI). Note that for the moment, the program display the ROI only when you finish your selection.
To detail the ROI, it is the area the program will treat. This ROI will be crop and convert to digits.
When the ROI are selected, press enter touch.

After that, the program will go through all the documents in input folder, keeping the same ROI, convert the result to digit, and compare with the correct result.

The programm will output each document with a square for each result. A green square mean the answer is correct, and orange one mean it is not.
The programm also display the reference number: for example, if the assignment contain a 4 and the answer was a 7, the number 7 will be display in the box.

## Training

A script is available to train again the model. Note that a training session will erase the previous weights!

## Built With

* [Python 3](https://www.python.org) - Python langage
* [Numpy](https://numpy.org) - The fundamental package for scientific computing with Python
* [Opencv](https://opencv.org) - Computer Vision library, tools, and hardware.
* [TensorFlow](https://www.tensorflow.org) - End-to-end open source platform for machine learning

## Authors

* **Arthur LANG** - *Initial work* - [LangArthur](https://github.com/LangArthur)
* **Mădălina Aldea** - *Initial work* - [MadalinaAldea](https://github.com/MadalinaAldea)

