import numpy as np
import random

def stochGradDescent(data, rate, tradeoff):
    size = len(data[0]) - 1 # account for label
    epsilon = random.uniform(0.0001, 0.001)
    w = np.full(size, epsilon)
    stoppingThreshold = 0.001
    diff = 1

    epochCount = 0
    while diff > stoppingThreshold:
        currRate = rate/(1+epochCount)
        random.shuffle(data)
        example = data[random.randint(0, len(data)-1)]
        yi = example[0]
        xi = example[1:]
        if np.dot(w, xi)*yi <= 1:
            wi = (1-currRate)*w + currRate*tradeoff*yi*xi
        else:
            wi = (1-currRate)*w
        diff = abs(np.sum(wi-w))
        epochCount += 1
        w = wi
        
    return w