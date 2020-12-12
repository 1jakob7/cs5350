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

def extractLabels(data):
    labels = []
    for example in data:
        labels.append(example[0])
    return labels

def createVectorsFromMiscs(data, labels):
    od = OrderedDict() # ensure data is consistent
    for example in data:
        for attribute in example:
            od[attribute] = None
    attrs = list(od.keys())
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = attrs.index(data[i][j])
            data[i].append(1) # add bias
        data[i][:0] = [int(labels[i])]
    return np.array(data)

def createVectorsFromMiscsNoLabel(data):
    od = OrderedDict() # ensure data is consistent
    for example in data:
        for attribute in example:
            od[attribute] = None
    attrs = list(od.keys())
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = attrs.index(data[i][j])
            data[i].append(1) # add bias
    return np.array(data)

def recordAccuracy(data, weightList):
    total = len(data)
    correctCount = 0
    for example in data:
        trueLabel = example[0]
        if makePrediction(example, weightList) == trueLabel:
            correctCount += 1
    return correctCount / total

def makePrediction(example, weightList):
    negCount = 0
    posCount = 0
    for weight in weightList:
        guess = np.dot(example[1:], weight)
        if guess < 0:
            negCount += 1
        else:
            posCount += 1
    if negCount > posCount:
        return 0
    else:
        return 1

def makePredictionNoLabel(example, weight):
    negCount = 0
    posCount = 0
    for weight in weightList:
        guess = np.dot(example, weight)
        if guess < 0:
            negCount += 1
        else:
            posCount += 1
    if negCount > posCount:
        return 0
    else:
        return 1

# Main
# setup global constants...
random.seed(17)
basePath = '../project_data/data/'

# read initial data
# glove data necessary for label extraction
trainGloveData = readFile(basePath + 'glove/glove.train.libsvm')
testGloveData = readFile(basePath + 'glove/glove.test.libsvm')
trainMiscData = readMiscFile(basePath + 'misc-attributes/misc-attributes-train.csv')
testMiscData = readMiscFile(basePath + 'misc-attributes/misc-attributes-test.csv')
evalMiscData = readMiscFile(basePath + 'misc-attributes/misc-attributes-eval.csv')
# extract labels and create vectors from misc data
trainLabels = extractLabels(trainGloveData)
trainMiscData = createVectorsFromMiscs(trainMiscData, trainLabels)
testLabels = extractLabels(testGloveData)
testMiscData = createVectorsFromMiscs(testMiscData, testLabels)

evalMiscData = createVectorsFromMiscsNoLabel(evalMiscData)

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
# initialLearningRates = [10, 1, 0.1, 0.01, 0.001]
# regTradeoffs = [10000, 1000, 100, 10, 1]
# epochs = [1000, 1500, 2000, 2500, 3000]

initialLearningRates = [0.1]
regTradeoffs = [100]
epochs = [2500]
bagCount = 5

print('Logistic regression (stochastic sub-gradient descent) w/ bagging:')
print('Hyper-parameters: learning rate = 0.1, tradeoff = 100, epochs = 2500, bags = 17')
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
                weightList = []
                for i in range(bagCount):
                    weightList.append(lr.stochSubGradDescent(trainingFolds, rate, tradeoff, epoch))
                # test
                accuracySum += recordAccuracy(testFold, weightList)
            print('Cross-validation accuracy: ' + str(accuracySum / foldCount))

# train on full dataset
weightList = []
for i in range(bagCount):
    weightList.append(lr.stochSubGradDescent(trainMiscData, 
        initialLearningRates[0], regTradeoffs[0], epochs[0]))
# test on training set
trainAccuracy = recordAccuracy(trainMiscData, weightList)
print('Train accuracy: ' + str(trainAccuracy))
# test on test set
testAccuracy = recordAccuracy(testMiscData, weightList)
print('Test accuracy: ' + str(testAccuracy))
# test on eval data and record results
# with open('log_reg_bag_predictions.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['example_id', 'label'])
#     for i in range(len(evalMiscData)):
#         result = makePredictionNoLabel(evalMiscData[i], weightList)
#         writer.writerow([str(i), result])
