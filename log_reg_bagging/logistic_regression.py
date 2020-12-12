import random
import numpy as np

def stochSubGradDescent(data, rate, tradeoff, epochs):
    size = len(data[0]) - 1 # account for label
    epsilon = random.uniform(0.00001, 0.0001)
    w = np.full(size, epsilon)

    for i in range(epochs):
        currRate = rate/(1+i)
        rand = random.randint(0, len(data) - 1)
        example = data[rand]
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