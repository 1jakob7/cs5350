import csv
import random
import numpy as np
from collections import OrderedDict

from sklearn.ensemble import RandomForestClassifier

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

def createVectorsFromMiscs(data):
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

def recordAccuracy(data, labels, classifier):
    total = len(data)
    correctCount = 0
    for i in range(len(data)):
        trueLabel = labels[i]
        prediction = classifier.predict(data[i].reshape(1, -1))[0]
        if prediction == trueLabel:
            correctCount += 1
    return correctCount / total

# Main
# setup global constants...
random.seed(17)
basePath = 'project_data/data/'

# read initial data
# glove data necessary for label extraction
trainGloveData = readFile(basePath + 'glove/glove.train.libsvm')
testGloveData = readFile(basePath + 'glove/glove.test.libsvm')
trainMiscData = readMiscFile(basePath + 'misc-attributes/misc-attributes-train.csv')
testMiscData = readMiscFile(basePath + 'misc-attributes/misc-attributes-test.csv')
evalMiscData = readMiscFile(basePath + 'misc-attributes/misc-attributes-eval.csv')
# create vectors from misc data
trainMiscData = createVectorsFromMiscs(trainMiscData)
testMiscData = createVectorsFromMiscs(testMiscData)
# extract labels
trainLabels = extractLabels(trainGloveData)
testLabels = extractLabels(testGloveData)

evalMiscData = createVectorsFromMiscs(evalMiscData)

# setup 5-fold cross-validation data
foldCount = 5
examplesPerFold = int(len(trainMiscData) / foldCount)
folds = {}
labelFolds = {}
for i in range(foldCount):
    fold = []
    labelFold = []
    startIndex = i * examplesPerFold
    endIndex = startIndex + examplesPerFold
    for j in range(startIndex, endIndex):
        fold.append(trainMiscData[j])
        labelFold.append(trainLabels[j])
    folds[i] = fold
    labelFolds[i] = labelFold

# hyper-parameters
# maxDepths = [1, 2, 3, 4, 5]
# maxSamples = [0.1, 0.2, 0.3, 0.4, 0.5]

maxDepths = [1]
maxSamples = [0.1]

print('Random forests')
print('Hyper-parameters: max depth = 1, max samples = 10%')
foldCount = 5 
for depth in maxDepths:
    for samples in maxSamples:
        # initialize scikit random forests classifier
        accuracySum = 0
        classifier = RandomForestClassifier(max_depth=depth)
        for k in range(foldCount):
            # assign test and training data
            testFold = folds[k]
            testLabelFold = labelFolds[k]
            trainingFolds = []
            trainLabelFolds = []
            for i in range(foldCount):
                if i != k:
                    trainingFolds += folds[i]
                    trainLabelFolds += labelFolds[i]
            # train
            classifier.fit(trainingFolds, trainLabelFolds)
            # test
            accuracySum += recordAccuracy(testFold, testLabelFold, classifier)
        print('Cross-validation accuracy: ' + str(accuracySum / foldCount))

classifier = RandomForestClassifier(max_depth=maxDepths[0], max_samples=maxSamples[0])
# full training set
classifier.fit(trainMiscData, trainLabels)
trainAccuracy = recordAccuracy(trainMiscData, trainLabels, classifier)
print('Training accuracy: ' + str(trainAccuracy))
# full test set
testAccuracy = recordAccuracy(testMiscData, testLabels, classifier)
print('Test accuracy: ' + str(testAccuracy))

# record eval results
# with open('random_forests.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['example_id', 'label'])
#     for i in range(len(evalMiscData)):
#         prediction = classifier.predict(evalMiscData[i].reshape(1, -1))[0]
#         writer.writerow([str(i), prediction])
    