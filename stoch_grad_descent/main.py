import csv
import random
import numpy as np
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
    vector = [0] * (attributeCount + 1) # adjust for label
    if example[0] == '0': # label
        vector[0] = -1 # want {-1, 1} for sgd alg
    else:
        vector[0] = 1
    attributes = example[1:len(example)]
    for attribute in attributes:
        attr = attribute.split(':')
        index = int(attr[0])
        vector[index] = float(attr[1])
    return vector

# def confineValues(data):
#     maxVal = 0
#     for example in data:
#         for attribute in example:
#             if abs(attribute) > abs(maxVal):
#                 maxVal = abs(attribute)
#     size = len(example)
#     for example in data:
#         for i in range(1, size): # adjust for label
#             example[i] = example[i] / maxVal

def getAccuracyModi(data, weights):
    correctCount = 0
    for example in data:
        trueLabel = example[0]
        if makePrediction(example, weights) == trueLabel:
            correctCount += 1
    return correctCount / len(data)

def makePrediction(example, weights):
    negVotes = 0
    posVotes = 0
    x = example[1:len(example)]
    for w in weights:
        guess = np.dot(w, x)
        if guess < 0:
            negVotes += 1
        else:
            posVotes += 1
    if negVotes > posVotes:
        return -1
    else:
        return 1

# Main
# setup constants
random.seed(17)
epochCount = 3
bagCount = 17
#initialLearningRates = [1, 0.1, 0.01, 0.001]
initialLearningRates = [0.01, 0.001, 0.0001, 0.00001]
#tradeoffs = [0.5, 0.1, 0.01, 0.001]
tradeoffs = [0.01, 0.001, 0.0001, 0.00001]

# setup training data - 'tfidf'
basePath = 'project_data/data/tfidf/'
trainData = readFile(basePath + 'tfidf.train.libsvm')

# reformat training data (filling in missing values
# with 0's is notable for report)
attributeCount = getNumberOfAttributes(trainData)
for i in range(len(trainData)):
    trainData[i] = createVector(trainData[i], attributeCount)
# preprocess attribute values (not necessary for 'tfidf' dataset)
# confineValues(trainData)

# train svm w/ stochastic gradient descent and bagging (whole dataset)
weightVectors = []
bagExampleCount = int(len(trainData) / 7)
for i in range(bagCount):
    trainingBag = random.sample(trainData, bagExampleCount)
    weightVectors.append(svm.stochGradDescent(
        trainingBag, initialLearningRates[0], tradeoffs[2], epochCount))
trainAccuracy = getAccuracyModi(trainData, weightVectors)
print('Training accuracy: ' + str(trainAccuracy))
# run classifier against test set
testData = readFile(basePath + 'tfidf.test.libsvm')
for i in range(len(testData)):
    testData[i] = createVector(testData[i], attributeCount)
testAccuracy = getAccuracyModi(testData, weightVectors)
print('Test accuracy: ' + str(testAccuracy))
# run classifier against eval set and record predictions
evalData = readFile(basePath + 'tfidf.eval.anon.libsvm')
for i in range(len(evalData)):
    evalData[i] = createVector(evalData[i], attributeCount)
with open('svm_stoch_grad_descent.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['example_id', 'label'])
    for i in range(len(evalData)):
        result = makePrediction(evalData[i], weightVectors)
        if result == -1:
            result = 0
        writer.writerow([str(i), result])


# 5-fold cross-validation
# foldCount = 5
# examplesPerFold = int(len(trainData) / foldCount)
# folds = {}
# for i in range(foldCount):
#     fold = []
#     startIndex = i * examplesPerFold
#     endIndex = startIndex + examplesPerFold
#     for j in range(startIndex, endIndex):
#         fold.append(trainData[j])
#     folds[i] = fold

# for tradeoff in tradeoffs:
#     avgAccuracies = []
#     for rate in initialLearningRates:
#         accuracySum = 0
#         for k in range(len(folds)):
#             # assign test and training data
#             testFold = folds[k]
#             trainingFolds = []
#             for i in range(foldCount):
#                 if i != k:
#                     trainingFolds = trainingFolds + folds[i]
#             # train svm
#             weightVectors = []
#             bagExampleCount = int(len(trainingFolds) / 7)
#             for i in range(bagCount):
#                 trainingBag = random.sample(trainingFolds, bagExampleCount)
#                 weightVectors.append(svm.stochGradDescent(
#                     trainingBag, rate, tradeoff, epochCount))
#             # weightVector = svm.stochGradDescent(
#             #     trainingFolds, rate, tradeoff, epochCount)

#             # test svm
#             accuracySum += getAccuracyModi(testFold, weightVectors)
#             #accuracySum += getAccuracy(testFold, weightVector)
#         avgAccuracies.append(accuracySum / foldCount)
#     # print results...
#     print('Epoch count: ' + str(epochCount) + 
#         ' Tradeoff constant: ' + str(tradeoff))
#     for i in range(len(avgAccuracies)):
#         print('Initial learning rate: ' + str(initialLearningRates[i])
#             + '\tAverage accuracy: ' + str(avgAccuracies[i]))
#     print()

