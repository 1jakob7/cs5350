import numpy as np
from os import listdir
import random
import svm

def readFile(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            example = line.split()
            data.append(example)
    return data

def getNumberOfAttributes(data):
    max = 0
    for example in data:
        for attribute in example:
            attrNum = int(attribute.split(':')[0])
            if attrNum > max:
                max = attrNum
    return max

def createVector(example, attributeCount):
    vector = np.zeros(attributeCount + 2) # adjust for label & bias
    vector[0] = int(example[0]) # label
    attributes = example[1:]
    for attribute in attributes:
        attr = attribute.split(':')
        index = int(attr[0])
        vector[index] = float(attr[1])
    vector[-1] = 1.0 # store bias at last index
    return vector

def makePrediction(example, weight):
    guess = np.dot(example[1:], weight)
    if guess < 0:
        return -1
    else:
        return 1

# Main
# setup global constants
random.seed(17)
basePath = 'data/libSVM-format/'
foldPath = basePath + 'CVFolds/'
# read initial data used by all algorithms
trainData = readFile(basePath + 'train')
testData = readFile(basePath + 'test')
attributeCount = getNumberOfAttributes(trainData)
# read fold data and reformat
folds = []
for fileName in listdir(foldPath):
    folds.append(readFile(foldPath + fileName))
numFolds = len(folds)
# reformat data
for fold in folds:
    for i in range(len(fold)):
        fold[i] = createVector(fold[i], attributeCount)

# setup svm constants
initialLearningRates = [1, 0.1, 0.01, 0.001, 0.0001]
regTradeoffs = [1000, 100, 10, 1, 0.1, 0.01]

# 5-fold cross-validation
for tradeoff in regTradeoffs:
    for rate in initialLearningRates:
        for k in range(numFolds):
            # assign test and training data
            testFold = folds[k]
            trainingFolds = folds[:k] + folds[k+1:]
            # train svm
            weight = svm.stochGradDescent(
                trainingFolds, rate, tradeoff)



# setup logistic regression constants