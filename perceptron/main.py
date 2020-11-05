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

# *not necessary for 'glove' data*
def getNumberOfAttributes(data):
    max = 0
    for example in data:
        for attribute in example:
            aNum = int(attribute.split(':')[0])
            if aNum > max:
                max = aNum
    return max

# *not necessary for 'glove' data*
# creates a full sized vector based on example's attributes
def createVector(example, attributeCount):
    vector = [0] * (attributeCount + 1)  # adjust for label
    if example[0] == '0': # label
        vector[0] = -1
    else:
        vector[0] = 1
    e = example[1:len(example)]
    for attribute in e: # attributes
        attr = attribute.split(':')
        index = int(attr[0])
        vector[index] = float(attr[1])
    return vector

# not necessary for non-'glove' data - vector creation
# already handles this
def removeAttributeIndexes(data, size):
    for example in data:
        for i in range(len(example)):
            if i > 0: # attribute
                sp = example[i].split(':')
                example[i] = float(sp[1])
            else: # label
                if example[i] == '0':
                    example[i] = -1
                else:
                    example[i] = 1


def recordAccuracy(data, w):
    size = len(data[0])
    correctCount = 0
    for example in data:
        trueLabel = example[0]
        x = example[1:size]
        guess = np.dot(w, x)
        if trueLabel == -1 and guess < 0:
            correctCount += 1
        elif trueLabel == 1 and guess >= 0:
            correctCount += 1
    return correctCount / len(data)

# Main
# setup constants
random.seed(42)
epochCount = 2
learningRates = [1, 0.1, 0.01, 0.001]

# setup training data - 'glove'
# basePath = 'project_data/data/glove/'
# trainData = readFile(basePath + 'glove.train.libsvm')

# setup training data = 'tfidf'
basePath = 'project_data/data/tfidf/'
trainData = readFile(basePath + 'tfidf.train.libsvm')

# get number of attributes and reformat training data
attributeCount = getNumberOfAttributes(trainData)
for i in range(len(trainData)):
    trainData[i] = createVector(trainData[i], attributeCount)

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
print('With an epoch count of ' + str(epochCount) + '...')
for i in range(len(avgAccuracies)):
    print('Learning rate: ' + str(learningRates[i]) + '\tAverage accuracy: '
    + str(avgAccuracies[i]) + '\tAverage updates: ' + str(avgUpdates[i]))
