import os

def readFile(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            example = line.split()
            data.append(example)
    return data

def getAttributeRange(data):
    highest = 0
    current = None
    for example in data:
        for attribute in example:
            current = int(attribute.split(':')[0])
            if current > highest:
                highest = current
    return [*range(1, highest + 1)]

# Main
# setup training data
basePath = 'project_data/data/'
trainData = readFile(basePath + 'bag-of-words/bow.train.libsvm')
attributes = getAttributeRange(trainData)

# 5-fold cross-validation to test hyper-parameter: depth
foldCount = 5
examplesPerFold = int(len(trainData) / foldCount)
folds = []
for i in range(foldCount):
    fold = []
    for j in range(examplesPerFold):
        fold.append(trainData[j])
    folds.append(fold)

bestDepth = 0
greatestAccuracy = 0
for k in range(10): # depth (1-10)
    k += 1
    accuracies = []
    for i in range(foldCount):
        # split data into training set and test set
        testFold = folds[i]
        trainingFolds = []
        for j in range(foldCount):
            if j != i:
                trainingFolds = trainingFolds + folds[j]
        root = DepthRestrictedID3(trainingFolds, attributes, 0, k)

        # test tree against withheld set
        count = 0
        correct = 0
        for example in testFold:
            label = int(example[0])
            result = makePrediction(root, example)
            if result == label:
                correct += 1
            count += 1
        accuracies.append(correct / count)