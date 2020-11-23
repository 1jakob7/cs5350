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

def makePrediction(example, w):
    x = example[1:len(example)]
    guess = np.dot(w, x)
    if guess < 0:
        return 0 # negative prediction
    else:
        return 1 # positive prediction

# Main
# setup constants
random.seed(42)
epochCount = 5
learningRates = [1, 0.1, 0.01, 0.001]

# setup training data - 'glove'
# basePath = 'project_data/data/glove/'
# trainData = readFile(basePath + 'glove.train.libsvm')

# setup training data - 'tfidf'
basePath = 'project_data/data/tfidf/'
trainData = readFile(basePath + 'tfidf.train.libsvm')

# get number of attributes and reformat training data
attributeCount = getNumberOfAttributes(trainData)
for i in range(len(trainData)):
    trainData[i] = createVector(trainData[i], attributeCount)
# train averaged perceptron
averagedResult = ap.averagedPerceptron(
    trainData, epochCount, learningRates[0])
print('Performed ' + str(averagedResult[1]) + ' updates.')

# setup test data and record accuracy
testData = readFile(basePath + 'tfidf.test.libsvm')
for i in range(len(testData)):
    testData[i] = createVector(testData[i], attributeCount)
testAccuracy = recordAccuracy(testData, averagedResult[0])
print('Test accuracy: ' + str(testAccuracy))

# setup eval data and record predictions
# evalData = readFile(basePath + 'tfidf.eval.anon.libsvm')
# with open('avg_perceptron_predictions.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['example_id', 'label'])
#     for i in range(len(evalData)):
#         evalData[i] = createVector(evalData[i], attributeCount)
#         result = makePrediction(evalData[i], averagedResult[0])
#         writer.writerow([str(i), result])


# # 5-fold cross-validation to test hyper-parameter: learningRate
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

# avgAccuracies = []
# avgUpdates = []
# for rate in learningRates:
#     accuracySum = 0
#     updatesSum = 0
#     for k in range(len(folds)):
#         # assign test and training data
#         testFold = folds[k]
#         trainingFolds = []
#         for i in range(foldCount):
#             if i != k:
#                 trainingFolds = trainingFolds + folds[i]
#         # train the perceptron, result holds the averaged
#         # weight vector and the number of updates performed
#         averagedResult = ap.averagedPerceptron(
#             trainingFolds, epochCount, rate)
#         accuracySum += recordAccuracy(
#             testFold, averagedResult[0])
#         updatesSum += averagedResult[1]
#     avgAccuracies.append(accuracySum / foldCount)
#     avgUpdates.append(updatesSum / foldCount)

# # print results...
# print('With an epoch count of ' + str(epochCount) + '...')
# for i in range(len(avgAccuracies)):
#     print('Learning rate: ' + str(learningRates[i]) + '\tAverage accuracy: '
#     + str(avgAccuracies[i]) + '\tAverage updates: ' + str(avgUpdates[i]))
