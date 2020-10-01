# Author: Jakob Horvath, u1092049

import random
import perceptrons
from os import listdir

def readFileSVM(path):
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
            aNum = int(attribute.split(':')[0])
            if aNum > max:
                max = aNum

    return max

# creates a full sized vector based on example's attributes
def createVector(example, attributeCount):
    vector = [0] * (attributeCount + 1)  # adjust for label
    vector[0] = int(example[0])
    e = example[1:len(example)]
    for attribute in e:
        index = int(attribute.split(':')[0])
        vector[index] = 1.0

    return vector

def getBestWeight(accuracies):
    bestWeight = []
    bestAccuracy = 0
    for accuracy in accuracies:
        if accuracy[0] > bestAccuracy:
            bestAccuracy = accuracy[0]
            bestWeight = accuracy[1]

    return bestWeight

def getMajorityBaselineAccuracy(data):
    posCount = 0
    negCount = 0
    correctCount = 0
    for example in data:
        guess = 1 if posCount > negCount else -1
        if guess == int(example[0]):
            correctCount += 1
        if example[0] == -1:
            negCount += 1
        else:
            posCount += 1
    
    return correctCount / len(data)

# *program starts here*
# set some program constants
random.seed(17)
trainEpochCount = 20
learningRates = [1, 0.1, 0.01]

# compile training data and get number of contained attributes
trainData = readFileSVM('data/libSVM-format/train')
attributeCount = getNumberOfAttributes(trainData)
fixedTrainData = []
for example in trainData:
    fixedTrainData.append(createVector(example, attributeCount))

# compile testing data
testData = readFileSVM('data/libSVM-format/test')
fixedTestData = []
for example in testData:
    fixedTestData.append(createVector(example, attributeCount))

# generate majority baselines
print('Training set accuracy for Majority Baseline: ' + str(getMajorityBaselineAccuracy(trainData)))
print('Test set accuracy for Majority Baseline: ' + str(getMajorityBaselineAccuracy(testData)) + '\n')

# k-fold using all perceptron algorithms
foldsEpochCount = 10

dir = 'data/libSVM-format/CVfolds/'
folds = []
count = 0
for fileName in listdir(dir):
    folds.append(readFileSVM(dir + fileName))

numFolds = len(folds)
simpleMeanAccuracies = []
decayMeanAccuracies = []
averagedMeanAccuracies = []
for rate in learningRates:
    simpleAccuracySum = 0
    decayAccuracySum = 0
    averagedAccuracySum = 0
    for k in range(numFolds):
        # assign training folds data
        trainFolds = []
        # combine training folds into a single set
        for i in range(numFolds):
            if i != k:
                trainFolds = trainFolds + folds[i]
        # new count for this unique training set
        attributeCount = getNumberOfAttributes(trainFolds)
        fixedTrainFolds = []
        for example in trainFolds:
            fixedTrainFolds.append(createVector(example, attributeCount))

        # assign testing fold data
        fixedTestFold = []
        for example in folds[k]:
            fixedTestFold.append(createVector(example, attributeCount))

        # train Simple Perceptron and test
        accuracies = perceptrons.simplePerceptron(
            fixedTrainFolds, foldsEpochCount, rate)
        bestWeight = getBestWeight(accuracies[0])
        simpleAccuracySum += perceptrons.recordAccuracy(fixedTestFold, bestWeight)
        # train Decaying Perceptron and test
        accuracies = perceptrons.decayingPerceptron(
            fixedTrainFolds, foldsEpochCount, rate)
        bestWeight = getBestWeight(accuracies[0])
        decayAccuracySum += perceptrons.recordAccuracy(fixedTestFold, bestWeight)
        # train Averaged Perceptron and test
        averagedWeight = perceptrons.averagedPerceptron(
            fixedTrainFolds, foldsEpochCount, rate)
        averagedAccuracySum += perceptrons.recordAccuracy(
            fixedTestFold, averagedWeight[0])
    simpleMeanAccuracies.append(simpleAccuracySum / numFolds)
    decayMeanAccuracies.append(decayAccuracySum / numFolds)
    averagedMeanAccuracies.append(averagedAccuracySum / numFolds)
# report best hyper-parameter (learning rate) and its accuracy for each perceptron
bestIndex = 0
bestAccuracy = 0
for i in range(len(simpleMeanAccuracies)):
    if (simpleMeanAccuracies[i] > bestAccuracy):
        bestAccuracy = simpleMeanAccuracies[i]
        bestIndex = i
print('Best hyper-parameter (learning rate) for Simple Perceptron: ' 
    + str(learningRates[bestIndex]))
print('Corresponding cross-validation accuracy: ' + str(bestAccuracy) + '\n')
simpleLearningRate = learningRates[bestIndex] #
bestIndex = 0
bestAccuracy = 0
for i in range(len(decayMeanAccuracies)):
    if (decayMeanAccuracies[i] > bestAccuracy):
        bestAccuracy = decayMeanAccuracies[i]
        bestIndex = i
print('Best hyper-parameter (learning rate) for Decaying Perceptron: ' 
    + str(learningRates[bestIndex]))
print('Corresponding cross-validation accuracy: ' + str(bestAccuracy) + '\n')
decayLearningRate = learningRates[bestIndex] #
bestIndex = 0
bestAccuracy = 0
for i in range(len(averagedMeanAccuracies)):
    if (averagedMeanAccuracies[i] > bestAccuracy):
        bestAccuracy = averagedMeanAccuracies[i]
        bestIndex = i
print('Best hyper-parameter (learning rate) for Averaged Perceptron: ' 
    + str(learningRates[bestIndex]))
print('Corresponding cross-validation accuracy: ' + str(bestAccuracy) + '\n')
averagedLearningRate = learningRates[bestIndex] #

# train Simple Perceptron and then test using most accurate weight vector
accuracies = perceptrons.simplePerceptron(
    fixedTrainData, trainEpochCount, simpleLearningRate)
bestWeight = getBestWeight(accuracies[0])
trainResult = perceptrons.recordAccuracy(fixedTrainData, bestWeight)
testResult = perceptrons.recordAccuracy(fixedTestData, bestWeight)
print('Simple Perceptron using best determined learning rate: ' + str(simpleLearningRate) + ' ...')
print('Updates performed: ' + str(accuracies[1]))
print('Training set accuracy: ' + str(trainResult))
print('Test set accuracy: ' + str(testResult) + '\n')

# train Decaying Perceptron and then test using most accurate weight vector
accuracies = perceptrons.decayingPerceptron(
    fixedTrainData, trainEpochCount, decayLearningRate)
bestWeight = getBestWeight(accuracies[0])
trainResult = perceptrons.recordAccuracy(fixedTrainData, bestWeight)
testResult = perceptrons.recordAccuracy(fixedTestData, bestWeight)
print('Decaying Perceptron using best determined learning rate: ' + str(decayLearningRate) + ' ...')
print('Updates performed: ' + str(accuracies[1]))
print('Training set accuracy: ' + str(trainResult))
print('Test set accuracy: ' + str(testResult) + '\n')

# train Averaged Perceptron and then test using averaged weight vector
averagedWeight = perceptrons.averagedPerceptron(
    fixedTrainData, trainEpochCount, averagedLearningRate)
trainResult = perceptrons.recordAccuracy(fixedTrainData, averagedWeight[0])
testResult = perceptrons.recordAccuracy(fixedTestData, averagedWeight[0])
print('Averaged Perceptron using best determined learning rate: ' + str(averagedLearningRate) + ' ...')
print('Updates performed: ' + str(averagedWeight[1]))
print('Training set accuracy: ' + str(trainResult))
print('Test set accuracy: ' + str(testResult))