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
    updates = 0
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
                updates += 1
        accuracies.append([recordAccuracy(data, w), w])

    return [accuracies, updates]

def decayingPerceptron(data, epochCount, learningRate):
    rand = random.uniform(-0.01, 0.01)
    size = len(data[0])
    accuracies = []
    updates = 0
    # initialize decay rate
    decayRate = 0
    decayIncrease = random.uniform(-0.001, 0.001)
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
                updates += 1
        accuracies.append([recordAccuracy(data, w), w])
        # increase decay and then apply to learning rate
        decayRate += abs(decayIncrease)
        learningRate = learningRate / (1 + decayRate)

    return [accuracies, updates]

def averagedPerceptron(data, epochCount, learningRate):
    rand = random.uniform(-0.01, 0.01)
    size = len(data[0])
    updates = 0
    # initialize weight vector and average weight vector
    w = []
    a = []
    for i in range(size - 1):
        w.append(rand)
        a.append(rand)
    # initialize bias term and average bias term
    b = rand
    ba = rand
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
                updates += 1
            # update the average weight and average bias regardless
            for j in range(size - 1):
                a[j] += w[j]
            ba += b

    return [a, updates]