import csv
import random
import numpy as np
import logistic_regression as lr
from collections import OrderedDict

def readFile(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            example = line.split()
            data.append(example)
    return data

def extractLabels(data):
    labels = []
    for example in data:
        labels.append(example[0])
    return labels

def readMiscFile(path):
    data = []
    with open(path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        first = True
        for row in csv_reader:
            if not first:
                for i in range(len(row)):
                    row[i] = str(row[i])
                data.append(row)
            else:
                first = False
    return data

def createVectorsFromMiscs(data, labels):
    od = OrderedDict() # ensure data is consistent
    for example in data:
        for attribute in example:
            od[attribute] = None
    attrs = list(od.keys())
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = attrs.index(data[i][j])
        data[i][:0] = [int(labels[i])]
    return np.array(data)

def getAttributes(data):
    max = 0
    for example in data:
        if (len(example) > max):
            max = len(example)
    return [*range(1, max)]

def recordAccuracy(data, weight):
    total = len(data)
    correctCount = 0
    for example in data:
        trueLabel = example[0]
        if makePrediction(example, weight) == trueLabel:
            correctCount += 1
    return correctCount / total

def makePrediction(example, weight):
    guess = np.dot(example[1:], weight)
    if guess < 0:
        return 0
    else:
        return 1


# Main
# setup global constants...
random.seed(17)
basePath = 'project_data/data/'

# read initial data
trainData = readFile(basePath + 'glove/glove.train.libsvm')
trainMiscData = readMiscFile(basePath + 'misc-attributes/misc-attributes-train.csv')
# ...

labels = extractLabels(trainData)
trainMiscData = createVectorsFromMiscs(trainMiscData, labels)
#attributes = getAttributes(trainMiscData)

# setup 5-fold cross-validation data
foldCount = 5
examplesPerFold = int(len(trainMiscData) / foldCount)
folds = {}
for i in range(foldCount):
    fold = []
    startIndex = i * examplesPerFold
    endIndex = startIndex + examplesPerFold
    for j in range(startIndex, endIndex):
        fold.append(trainMiscData[j])
    folds[i] = fold

# hyper-params to test for logisitic regression
initialLearningRates = [10, 1, 0.1, 0.01, 0.001]
regTradeoffs = [10000, 1000, 100, 10, 1]
epochs = [300, 275, 250, 225, 200]

for epoch in epochs:
    for tradeoff in regTradeoffs:
        for rate in initialLearningRates:
            accuracySum = 0
            for k in range(foldCount):
                # assign test and training data
                testFold = folds[k]
                trainingFolds = []
                for i in range(foldCount):
                    if i != k:
                        trainingFolds += folds[i]
                # train
                weight = lr.stochSubGradDescent(trainingFolds, rate, tradeoff, epoch)
                # test
                accuracySum += recordAccuracy(testFold, weight)
            print('Epochs: ' + str(epoch) + '\tTradeoff: ' + str(tradeoff) + '\tRate: ' + str(rate) +
            '\tCross-validation accuracy: ' + str(accuracySum / foldCount))
    print()

# voting system between log regs that read misc data and others
# that read glove or tfidf?