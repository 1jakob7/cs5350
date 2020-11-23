import numpy as np
import random

def stochGradDescent(data, rate, tradeoff):
    size = len(data[0]) - 1 # account for label
    epsilon = random.uniform(0.0001, 0.001)
    w = np.full(size, epsilon)
    stoppingThreshold = 0.009
    prevDiff = 0
    diff = 1

    epochCount = 0
    while abs(diff) > stoppingThreshold:
        currRate = rate/(1+epochCount)
        random.shuffle(data)
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
        
    return w