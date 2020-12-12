import random
import numpy as np

def stochGradDescent(data, learningRate, tradeoff, epochCount):
    vLength = len(data[0])
    weight = [0] * (vLength - 1) # ignore embedded label
    bias = 0
    for epoch in range(epochCount):
        #rate = learningRate / (1+epoch)
        rate = learningRate / (1 + (learningRate*epochCount)/tradeoff)
        random.shuffle(data)
        for example in data:
            yi = example[0]
            xi = example[1:vLength]
            if (yi * np.dot(weight, xi) + bias) <= 1: # error
                weight = (np.multiply((1-rate), weight) + 
                    np.multiply(rate*tradeoff*yi, xi))
            else:                                     # correct
                weight = (1-rate)*weight
            bias = bias + (yi * learningRate)
    return weight