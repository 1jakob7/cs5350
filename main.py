import numpy as np
from os import listdir
import random
import svm
import logistic_regression as lr
import decision_tree as dt

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

def recordAccuracy(data, weight):
    total = len(data)
    correctCount = 0
    for example in data:
        trueLabel = example[0]
        if makePrediction(example, weight) == trueLabel:
        #if makeLogPrediction(example, weight) == trueLabel:
            correctCount += 1
    return correctCount / total

def makePrediction(example, weight):
    guess = np.dot(example[1:], weight)
    if guess < 0:
        return -1
    else:
        return 1

def makeLogPrediction(example, weight):
    guess = 1 / (1 + np.exp(np.dot(example[1:], weight)))
    #guess = np.dot(example[1:], weight)
    if guess < 0.5:
        return -1
    else:
        return 1

def makeID3Prediction(example, root):
    while len(root.children) > 0:
        if example[root.value] == 1:
            root = root.children[1]
        else:
            root = root.children[-1]
    return float(root.value) # helps w/ svm

def mostFrequent(data):
    return max(set(data), key = data.count)

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

# 5-fold cross-validation for svm
for tradeoff in regTradeoffs:
    for rate in initialLearningRates:
        accuracySum = 0
        for k in range(numFolds):
            # assign test and training data
            testFold = folds[k]
            trainingFolds = []
            for i in range(numFolds):
                if i != k:
                    trainingFolds += folds[i]
            # train svm
            weight = svm.stochGradDescent(
                trainingFolds, rate, tradeoff)
            # test svm classifier
            accuracySum += recordAccuracy(testFold, weight)
        print('tradeoff: ' + str(tradeoff) + '\trate: ' + str(rate) + 
            '\taverage accuracy: ' + str(accuracySum / numFolds))
    print()

# setup logistic regression constants
initialLearningRates = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
regTradeoffs = [1000, 100, 10, 1, 0.1, 0.01]

# 5-fold cross-validation for logistic regression
for tradeoff in regTradeoffs:
    for rate in initialLearningRates:
        accuracySum = 0
        for k in range(numFolds):
            # assign test and training data
            testFold = folds[k]
            trainingFolds = []
            for i in range(numFolds):
                if i != k:
                    trainingFolds += folds[i]
            # train logistic regression
            weight = lr.stochGradDescent(
                trainingFolds, rate, tradeoff)
            # test logistic regression classifier
            accuracySum += recordAccuracy(testFold, weight)
        print('tradeoff: ' + str(tradeoff) + '\trate: ' + str(rate) + 
            '\taverage accuracy: ' + str(accuracySum / numFolds))
    print()

# setup svm over trees constants
initialLearningRates = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
regTradeoffs = [1000, 100, 10, 1, 0.1, 0.01]
depths = [1, 2, 4, 8]

numTrees = 200
sampleSize = 223
attributes = [*range(1, attributeCount)] # ignore the bias during ID3 alg
# 5-fold cross-validation for svm over trees
# for depth in depths:
#     for tradeoff in regTradeoffs:
#         for rate in initialLearningRates:
#             accuracySum = 0
#             for k in range(numFolds):
#                 # assign test and training data
#                 testFold = folds[k]
#                 trainingFolds = []
#                 for i in range(numFolds):
#                     if i != k:
#                         trainingFolds += folds[i]
#                 # train svm over trees -> trees first...
#                 trees = []
#                 for i in range(numTrees):      
#                     sample = random.choices(trainingFolds, k = sampleSize)
#                     trees.append(dt.ID3(sample, attributes, 0, depth))
#                 trainTransforms = []
#                 for example in trainingFolds:
#                     predictions = [example[0]] # start w/ true label
#                     for tree in trees:
#                         predictions.append(makeID3Prediction(example, tree))
#                     predictions.append(1.0) # *add bias
#                     trainTransforms.append(np.array(predictions))
#                 # ...now train the svm on the feature transformations
#                 weight = svm.stochGradDescent(trainTransforms, 1, 1000)
#                 # time for testing
#                 testTransforms = []
#                 for example in testFold:
#                     predictions = [example[0]] # start w/ true label
#                     for tree in trees:
#                         predictions.append(makeID3Prediction(example, tree))
#                     predictions.append(1.0) # *add bias
#                     testTransforms.append(np.array(predictions))
#                 accuracySum += recordAccuracy(testTransforms, weight)

#             print('depth ' + str(depth) + '\ttradeoff: ' + str(tradeoff) 
#             + '\trate: ' + str(rate) + '\taverage accuracy: ' + str(accuracySum / numFolds))
#         print()
                


