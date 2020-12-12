from decisionTree import depthRestrictedID3
import csv

def readFile(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            example = line.split()
            data.append(example)
    return data

# narrows down to the k-most frequent attributes
def getTrimmedAttributes(data, k):
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
    for i in range(k):
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
basePath = '../project_data/data/bag-of-words/'
trainData = readFile(basePath + 'bow.train.libsvm')
tAttributes = getTrimmedAttributes(trainData, 250) # take top 250 attributes

maxDepths = [68]

# 5-fold cross-validation to test hyper-parameter: depth
foldCount = 5
examplesPerFold = int(len(trainData) / foldCount)
folds = {}
for i in range(foldCount):
    fold = []
    startIndex = i * examplesPerFold
    endIndex = startIndex + examplesPerFold
    for j in range(startIndex, endIndex):
        fold.append(trainData[j])
    folds[i] = fold

print('Decision tree w/ limited depth')
print('Hyper-parameter: max depth = 68')

bestDepth = 0
greatestAccuracy = 0
for depth in maxDepths:
    accuracies = []
    for i in range(foldCount):
        # split data into training set and test set
        testFold = folds[i]
        trainingFolds = []
        for j in range(foldCount):
            if j != i:
                trainingFolds = trainingFolds + folds[j]
        root = depthRestrictedID3(trainingFolds, tAttributes, 0, depth)
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
    print('Cross-validation accuracy: ' + str(sum(accuracies) / len(accuracies)))
#print('\nBest Accuracy = ' + str(greatestAccuracy) + ' w/ Depth: ' + str(bestDepth))

root = depthRestrictedID3(trainData, tAttributes, 0, maxDepths[0]) # max-depth of 68 - found to be best
# train accuracy
count = 0
correct = 0
for example in trainData:
            label = example[0]
            result = makePrediction(root, example)
            if result == label:
                correct += 1
print('Training accuracy: ' + str(correct / count))

# setup test data
testData = readFile(basePath + 'bow.test.libsvm')
# test accuracy
count = 0
correct = 0
for example in testData:
            label = example[0]
            result = makePrediction(root, example)
            if result == label:
                correct += 1
            count += 1
print('Test accuracy: ' + str(correct / count))

# setup eval data
# evalData = readFile(basePath + 'bow.eval.anon.libsvm')
# # store eval predictions in .csv file
# with open('decision_tree_predictions.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['example_id', 'label'])
#     for i in range(len(evalData)):
#         result = makePrediction(root, evalData[i])
#         writer.writerow([str(i), result])
