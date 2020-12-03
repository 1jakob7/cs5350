import numpy as np
import random

def stochGradDescent(data, rate, tradeoff):
    size = len(data[0]) - 1 # account for label
    epsilon = random.uniform(0.0001, 0.001)
    w = np.full(size, epsilon)
    stoppingThreshold = 0.0001
    prevDiff = 0
    diff = 1

    lossList = []
    epochCount = 0
    while abs(diff) > stoppingThreshold:
        currRate = rate/(1+epochCount)
        example = data[random.randint(0, len(data)-1)]
        yi = example[0]
        xi = example[1:]
        diff = np.dot(w, xi)*yi
        if diff <= 1:
            w = (1-currRate)*w + currRate*tradeoff*yi*xi
        else:
            w = (1-currRate)*w
        diff = abs(diff - prevDiff)
        prevDiff = diff
        epochCount += 1
        #lossList.append(computeLoss(data, w))

    return w, lossList

def computeLoss(data, w):
    totalLoss = 0
    for example in data:
        yi = example[0]
        xi = example[1:]
        totalLoss += np.max([0, 1-yi*np.dot(w, xi)])

    return totalLoss
