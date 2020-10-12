from decisionTree import depthRestrictedID3

def readFile(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            example = line.split()
            data.append(example)
    return data

# narrows down to the 100 most frequent attributes - used
def getTrimmedAttributes(data):
    attributeCounts = {}
    for example in data:
        l = len(example)
        for i in range(1, l):
            att = example[i].split(':')[0]
            if att in attributeCounts:
                attributeCounts[att] += 1
            else:
                attributeCounts[att] = 1
    sortedAttributeCounts = sorted(attributeCounts.items(), 
        key = lambda x: x[1], reverse = True)
    commonAttributes = []
    for i in range(100):
        commonAttributes.append(sortedAttributeCounts[i][0])
    return commonAttributes

def makePrediction(root, example):
    while len(root.children) > 0:
        if (root.value + ':1') in example:
            root = root.children['1']
        else:
            root = root.children['0']

    return root.value

# Main
# setup training data
basePath = 'project_data/data/'
trainData = readFile(basePath + 'bag-of-words/bow.train.libsvm')
tAttributes = getTrimmedAttributes(trainData) # take top 100 attributes

# 5-fold cross-validation to test hyper-parameter: depth
foldCount = 5
examplesPerFold = int(len(trainData) / foldCount)
folds = {}
for i in range(foldCount):
    fold = []
    for j in range(examplesPerFold):
        fold.append(trainData[j])
    folds[i] = fold

bestDepth = 0
greatestAccuracy = 0
for k in range(50): # depth (1-10)
    k += 1
    accuracies = []
    for i in range(foldCount):
        # split data into training set and test set
        testFold = folds[i]
        trainingFolds = []
        for j in range(foldCount):
            if j != i:
                trainingFolds = trainingFolds + folds[j]
        root = depthRestrictedID3(trainingFolds, tAttributes, 0, k)
        # test tree against withheld set
        count = 0
        correct = 0
        for example in testFold:
            label = example[0]
            result = makePrediction(root, example)
            if result == label:
                correct += 1
            count += 1
        accuracies.append(correct / count)
    mean = sum(accuracies) / len(accuracies)
    if (mean > greatestAccuracy):
        greatestAccuracy = mean
        bestDepth = k
    print('Run Accuracy: ' + str(mean) + ' w/ Depth: ' + str(k))

print('\nBest Accuracy = ' + str(greatestAccuracy) + ' w/ Depth: ' + str(bestDepth))