import numpy as np
from os import listdir
import random
import svm
import logistic_regression as lr
import decision_tree as dt
import csv

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
            correctCount += 1
    return correctCount / total

def makePrediction(example, weight):
    guess = np.dot(example[1:], weight)
    if guess < 0:
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
# reformat data
for i in range(len(trainData)):
    trainData[i] = createVector(trainData[i], attributeCount)
for i in range(len(testData)):
    testData[i] = createVector(testData[i], attributeCount)
# read fold data and reformat
folds = []
for fileName in listdir(foldPath):
    folds.append(readFile(foldPath + fileName))
numFolds = len(folds)
# reformat data
for fold in folds:
    for i in range(len(fold)):
        fold[i] = createVector(fold[i], attributeCount)

# # setup svm constants
# initialLearningRates = [1, 0.1, 0.01, 0.001, 0.0001]
# regTradeoffs = [1000, 100, 10, 1, 0.1, 0.01]
# determined to be the optimal hyper-parameters
initialLearningRates = [1]
regTradeoffs = [1000]

print('SVM w/ learning rate = 1, tradeoff = 1000:')
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
                trainingFolds, rate, tradeoff)[0]
            # test svm classifier
            accuracySum += recordAccuracy(testFold, weight)
        print('Cross-validation accuracy: ' + str(accuracySum / numFolds))
# training on svm
weight, lossList = svm.stochGradDescent(trainData, initialLearningRates[0],
    regTradeoffs[0])
trainAccuracy = recordAccuracy(trainData, weight)
print('Training set accuracy: ' + str(trainAccuracy))
# store loss data in csv file
with open('svm_loss.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['epoch', 'loss'])
    for i in range(len(lossList)):
        writer.writerow([str(i), lossList[i]])
# test on svm
testAccuracy = recordAccuracy(testData, weight)
print('Test set accuracy: ' + str(testAccuracy) + '\n')

# setup logistic regression constants
# initialLearningRates = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
# regTradeoffs = [0.1, 1, 10, 100, 1000, 10000]
# determined to be the optimal hyper-parameters
initialLearningRates = [0.01]
regTradeoffs = [10000]

print('Logistic Regression w/ learning rate = 0.01, tradeoff = 10000:')
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
                trainingFolds, rate, tradeoff)[0]
            # test logistic regression classifier
            accuracySum += recordAccuracy(testFold, weight)
        print('Cross-validation accuracy: ' + str(accuracySum / numFolds))
# training on log reg
weight, lossList = lr.stochGradDescent(trainData, initialLearningRates[0],
    regTradeoffs[0])
trainAccuracy = recordAccuracy(trainData, weight)
print('Training set accuracy: ' + str(trainAccuracy))
# store loss data in csv file
with open('log_reg_loss.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['epoch', 'loss'])
    for i in range(len(lossList)):
        writer.writerow([str(i), lossList[i]])
# test on log reg
testAccuracy = recordAccuracy(testData, weight)
print('Test set accuracy: ' + str(testAccuracy) + '\n')

# # setup svm over trees constants
# initialLearningRates = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
# regTradeoffs = [1000, 100, 10, 1, 0.1, 0.01]
# depths = [1, 2, 4, 8]
# determined to be the optimal hyper-parameters
initialLearningRates = [0.001]
regTradeoffs = [1000]
depths = [8]

numTrees = 200
sampleSize = 223
attributes = [*range(1, attributeCount)] # ignore the bias during ID3 alg
print('SVM Over Trees w/ learning rate = 0.001, tradeoff = 1000, depth = 8:')
# 5-fold cross-validation for svm over trees
for depth in depths:
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
                # train svm over trees -> trees first...
                trees = []
                for i in range(numTrees):
                    sample = random.choices(trainingFolds, k = sampleSize)
                    trees.append(dt.ID3(sample, attributes, 0, depth))
                trainTransforms = []
                for example in trainingFolds:
                    predictions = [example[0]] # start w/ true label
                    for tree in trees:
                        predictions.append(makeID3Prediction(example, tree))
                    predictions.append(1.0) # *add bias
                    trainTransforms.append(np.array(predictions))
                # ...now train the svm on the feature transformations
                weight = svm.stochGradDescent(trainTransforms, 1, 1000)[0]
                # time for testing
                testTransforms = []
                for example in testFold:
                    predictions = [example[0]] # start w/ true label
                    for tree in trees:
                        predictions.append(makeID3Prediction(example, tree))
                    predictions.append(1.0) # *add bias
                    testTransforms.append(np.array(predictions))
                accuracySum += recordAccuracy(testTransforms, weight)
            print('Cross-validation accuracy: ' + str(accuracySum / numFolds))
# training on svm over trees
# trees first...
trees = []
for i in range(numTrees):
    sample = random.choices(trainData, k = sampleSize)
    trees.append(dt.ID3(sample, attributes, 0, depths[0]))
trainTransforms = []
for example in trainData:
    predictions = [example[0]] # start w/ true label
    for tree in trees:
        predictions.append(makeID3Prediction(example, tree))
    predictions.append(1.0) # *add bias
    trainTransforms.append(np.array(predictions))
# ...now svm on the trees
weight, lossList = svm.stochGradDescent(trainTransforms, initialLearningRates[0],
    regTradeoffs[0])
trainAccuracy = recordAccuracy(trainTransforms, weight)
print('Training set accuracy: ' + str(trainAccuracy))
# store loss data in csv file
with open('svm_over_trees_loss.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['epoch', 'loss'])
    for i in range(len(lossList)):
        writer.writerow([str(i), lossList[i]])
# test on svm over trees
# gotta convert to trees first...
trees = []
for i in range(numTrees):
    sample = random.choices(testData, k = sampleSize)
    trees.append(dt.ID3(sample, attributes, 0, depths[0]))
testTransforms = []
for example in testData:
    predictions = [example[0]] # start w/ true label
    for tree in trees:
        predictions.append(makeID3Prediction(example, tree))
    predictions.append(1.0) # *add bias
    testTransforms.append(np.array(predictions))
# ...now svm
testAccuracy = recordAccuracy(testTransforms, weight)
print('Test set accuracy: ' + str(testAccuracy))
