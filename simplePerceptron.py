### Author: Jakob Horvath, u1092049

import random
import numpy as np
import pandas as pd

def recordAccuracy(data, w):
    size = len(data[0])
    correctCount = 0
    for example in data:
        trueLabel = int(example[0])
        x = example[1:size]
        guess = np.dot(w, x)
        if trueLabel == -1 and guess < 0:
            correctCount += 1
        elif trueLabel == 1 and guess >= 0:
            correctCount += 1
    
    return correctCount / len(data)

def simplePerceptron(data, epochCount, learningRate):
    rand = random.uniform(-0.01, 0.01)
    size = len(data[0])
    accuracies = []
    # initialize weight vector
    w = []
    for i in range(size - 1):
        w.append(rand)
    # initialize the bias term
    b = rand
    # enter epoch loop
    for i in range(epochCount):
        random.shuffle(data)
        for example in data:
            yi = int(example[0]) # label
            xi = example[1:size] # attributes
            # update the weight and bias if there was an error
            if (yi * (np.dot(w, xi) + b)) <= 0:
                adjustment = np.multiply(np.array(xi), (yi * learningRate)) #.tolist()
                for j in range(len(w)):
                    w[j] = w[j] + adjustment[j]
                b = b + (yi * learningRate)
        accuracies.append([recordAccuracy(data, w), w])

    #return w
    return accuracies

### Main - will be its own file later
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
    vector = [0] * (attributeCount + 1) # adjust for label
    vector[0] = int(example[0])
    e = example[1:len(example)]
    for attribute in e:
        index = int(attribute.split(':')[0])
        vector[index] = 1.0
    return vector


random.seed(17)
trainEpochCount = 20
learningRate = [1, 0.1, 0.01]

trainData = readFileSVM('data/libSVM-format/train')
attributeCount = getNumberOfAttributes(trainData)
fixedTrainData = []
for example in trainData:
    fixedTrainData.append(createVector(example, attributeCount))

#w = simplePerceptron(fixedTrainData, trainEpochCount, learningRate[0])
accuracies = simplePerceptron(fixedTrainData, trainEpochCount, learningRate[0])


testData = readFileSVM('data/libSVM-format/test')
# ...