import random
import numpy as np
from math import exp

def stochGradDescent(data, rate, tradeoff):
    size = len(data[0]) - 1 # account for label
    totalEx = len(data)-1
    epsilon = random.uniform(0.0001, 0.001)
    w = np.full(size, epsilon)
    stoppingThreshold = 0.01
    prevDiff = 0
    diff = 1

    epochCount = 0
    while diff > stoppingThreshold:
        currRate = rate/(1+epochCount)
        random.shuffle(data)
        example = data[random.randint(0, totalEx)]
        yi = example[0]
        xi = example[1:]
        # compute gradient of 'log reg' objective function
        e = exp(-yi*np.dot(w, xi))
        diff = ((-yi*xi*e) / (1+e)) + ((2*w) / tradeoff)
        # update weight vector
        w = w - currRate*diff*w
        # update gradient difference (determines when to stop)
        diff = abs(diff - prevDiff)
        prevDiff = diff
        epochCount += 1
    
    return w