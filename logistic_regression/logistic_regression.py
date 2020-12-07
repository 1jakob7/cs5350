import random
import numpy as np

def stochGradDescent(data, rate, tradeoff, epochs):
    size = len(data[0]) - 1 # account for label
    epsilon = random.uniform(0.0001, 0.001)
    w = np.full(size, epsilon)

    #epochCount = 1
    for i in range(epochs):
        currRate = rate/(1+i)
        random.shuffle(data)
        for example in data:
            yi = example[0]
            # adjust for alg
            if yi == 0:
                yi = -1
            xi = example[1:]
            # careful w/ the exponential
            prod = -yi*np.dot(w, xi)
            if prod > 10:
                prod = 10
            elif prod < -10:
                prod = -10
            # compute the gradient
            grad = (-yi*xi*np.exp(prod)) / (1+np.exp(prod)) + (2*w)/tradeoff
            pred = np.dot(w, xi)*yi
            # update if incorrect prediction
            if pred <= 0:
                w = w - currRate*grad
    return w

def stochSubGradDescent(data, rate, tradeoff, epochs):
    size = len(data[0]) - 1 # account for label
    epsilon = random.uniform(0.0001, 0.001)
    w = np.full(size, epsilon)

    for i in range(epochs):
        currRate = rate/(1+i)
        batchData = random.sample(data, 1)
        for example in batchData:
            yi = example[0]
            # adjust for alg
            if yi == 0:
                yi = -1
            xi = example[1:]
            # careful w/ the exponential
            prod = -yi*np.dot(w, xi)
            if prod > 10:
                prod = 10
            elif prod < -10:
                prod = -10
            # compute the gradient
            grad = (-yi*xi*np.exp(prod)) / (1+np.exp(prod)) + (2*w)/tradeoff
            pred = np.dot(w, xi)*yi
            # update if incorrect prediction
            if pred <= 0:
                w = w - currRate*grad
    return w

def computeLoss(data, w):
    totalLoss = 0
    for example in data:
        yi = example[0]
        xi = example[1:]
        totalLoss += np.log(1 + np.exp(-yi*np.dot(w, xi)))

    return totalLoss