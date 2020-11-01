import csv
import random
import numpy as np
import avg_perceptron as ap

def readFile(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            example = line.split()
            data.append(example)
    return data

def recordAccuracy(data, w):
    size = len(data[0])
    correctCount = 0
    for example in data:
        trueLabel = int(example[0])
        x = example[1:size]
        guess = np.dot(w, x)
        if trueLabel == -1 and guess < 0:
            correctCount += 1
        elif trueLabel == 1 and guess >= 0:
            correctCount += 1
    return correctCount / len(data)

# Main
# setup constants
random.seed(17)
epochCount = 10
learningRates = [1, 0.1, 0.01]

# setup training data
basePath = 'project_data/data/bag-of-words/'
trainData = readFile(basePath + 'glove.train.libsvm')
attributeCount = 300 # sue me

# 5-fold cross-validation to test hyper-parameter: learningRate
foldCount = 5
examplesPerFold = int(len(trainData) / foldCount)
folds = {}
for i in range(foldCount):
    fold = []
    startIndex = i * examplesPerFold
    endIndex = startIndex + examplesPerFold
    for j in range(startIndex, endIndex):
        fold.append(trainData[j])
    folds[i] = fold

avgAccuracies = []
avgUpdates = []
for rate in learningRates:
    accuracySum = 0
    updatesSum = 0
    for k in range(len(folds)):
        # assign test and training data
        testFold = folds[k]
        trainingFolds = []
        for i in range(foldCount):
            if i != k:
                trainingFolds = trainingFolds + folds[i]
        # train the perceptron, result holds the averaged
        # weight vector and the number of updates performed
        averagedResult = ap.averagedPerceptron(
            trainingFolds, epochCount, rate)
        accuracySum += recordAccuracy( 
            testFold, averagedResult[0])
        updatesSum += averagedResult[1]
    avgAccuracies.append(accuracySum / foldCount)
    avgUpdates.append(updatesSum / foldCount)

# print results...

