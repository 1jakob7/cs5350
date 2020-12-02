import random
import numpy as np

def stochGradDescent(data, rate, tradeoff):
    #maxEx = len(data)*5
    size = len(data[0]) - 1 # account for label
    epsilon = random.uniform(0.0001, 0.001)
    w = np.full(size, epsilon)
    #stoppingThreshold = 0.001
    #gradDiff = 0
    grad = 1

    epochCount = 5
    for i in range(epochCount):
        currRate = rate/(1+i)
        random.shuffle(data)
        for example in data:
            yi = example[0]
            xi = example[1:]

            m = xi.shape[0]

            sigmoid = 1 / (1 + np.exp(np.dot(w, xi)))

            # cost = (-1/w)*(np.sum(yi*np.log(sigmoid)) + 
            #     ((1-yi)*(np.log(1-sigmoid))))
            
            grad = (1/m)*(np.dot(xi, (sigmoid-yi)))

            w = w - rate*grad

    return w


    # # compute gradient of 'log reg' objective function
    #         product = -yi*np.dot(xi, w)
    #         if product > 10:
    #             product = 10
    #         elif product < -10:
    #             product = -10
    #         e = np.exp(product)
    #         grad = (-yi*xi*e) / (1 + e) + (2*w / tradeoff)
    #         # update weight
    #         w = w - currRate*np.dot(grad, w)