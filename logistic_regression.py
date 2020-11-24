import random
import numpy as np
from math import exp

def stochGradDescent(data, rate, tradeoff):
    #maxEx = len(data)*5
    size = len(data[0]) - 1 # account for label
    epsilon = random.uniform(0.0001, 0.001)
    w = np.full(size, epsilon)
    stoppingThreshold = 0.01
    gradDiff = 0
    grad = 1

    #epochCount = 0
    #while diff > stoppingThreshold:
    epochCount = 10000
    for i in range(epochCount):
        currRate = rate/(1+i)
        random.shuffle(data)
        example = data[random.randint(0, len(data)-1)]
        yi = example[0]
        xi = example[1:]
        # compute gradient of 'log reg' objective function
        #e = exp(-yi*np.dot(w, xi))
        #grad = ((-yi*xi*e) / (1+e)) + ((2*w) / tradeoff)
        grad = (-yi*xi) + (2*w)/tradeoff
        # update weight vector
        w = w - currRate*np.dot(grad, w)
    
    return w