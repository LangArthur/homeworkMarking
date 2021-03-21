#!/usr/bin/env python3

import sys

from src.HomeworkMarking import *

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
