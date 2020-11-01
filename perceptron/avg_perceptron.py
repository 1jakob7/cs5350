import random
import numpy as np
import pandas as pd

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