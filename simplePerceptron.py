### Author: Jakob Horvath, u1092049

import random

def simplePerceptron(data):
    w = []
    rand = random.uniform(-0.01, 0.01)
    size = len(data[0]) - 1 # adjust for label
    for i in range(size):
        w.append(rand)
    print(w)

### Main - will be its own file later
def readFileSVM(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            example = line.split()
            data.append(example)
    
    return data


random.seed(17)

trainData = readFileSVM('data/libSVM-format/train')
simplePerceptron(trainData)

testData = readFileSVM('data/libSVM-format/test')
# ...