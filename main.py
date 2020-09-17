from math import sqrt
from os import listdir
from decisionTrees import get_common_label
from decisionTrees import get_overall_entropy
from decisionTrees import print_highest_info_gain
from decisionTrees import ID3
from decisionTrees import ID3_with_depth_restriction
from decisionTrees import test_tree

def read_file(path):
    data = []
    # open file for reading, "with" handles automatically closing the file
    with open(path, 'r') as file:
        for line in file:
            example = line.split()
            # store label (i.e. -1 or 1) along w/ associated feature vector as an array (<label>, <index>:<value>, ...)
            data.append(example)

    return data

def get_measured_attributes(examples):
    found_attributes = []
    for example in examples:
        for value in example:
            if ':' in value:
                attribute = value.split(':').pop(0)
                if attribute not in found_attributes:
                    found_attributes.append(value.split(':').pop(0))

    return found_attributes

# Main program
# setup and training data
examples = read_file('data/a1a.train')
attributes = get_measured_attributes(examples)
print('Most Common Label in Training Data: ' + get_common_label(examples) + '\n') ### PRINTa ###
print('Entropy of Training Data: ' + str(get_overall_entropy(examples)) + '\n') ### PRINTb ###
print_highest_info_gain(examples, attributes) ### PRINTc ###
root = ID3(examples, attributes)
#print('Depth: ' + str(root.get_depth()) + '\n')

count = 0
correct = 0
for example in examples:
    label = example[0]  # store for validation
    result = test_tree(root, example)
    if result == label:
        correct += 1
    count += 1

print('Training Set Accuracy: ' + str(correct / count) + '\n') ### PRINTd ###

# test data
tests = read_file('data/a1a.test')

count = 0
correct = 0
for test in tests:
    label = test[0]
    result = test_tree(root, test)
    if result == label:
        correct += 1
    count += 1

print('Test Set Accuracy: ' + str(correct / count) + '\n') ### PRINTe ###

# limiting depth w/ 5-fold cross-validation
dir = 'data/CVfolds/'
folds = {}
count = 0
for fileName in listdir(dir):
    foldData = read_file(dir + fileName)
    folds[count] = foldData
    count += 1

bestDepth = 0
greatestAccuracy = 0

for k in range(5):
    k += 1
    acccuracies = []
    for i in range(len(folds)):
        # split data into training set and test set
        testFold = folds[i]
        trainingFolds = []
        for j in range(len(folds)):
            if j != i:
                trainingFolds = trainingFolds + folds[j]

        # build depth-restricted tree (k-max) from training set
        attributes = get_measured_attributes(trainingFolds)
        root = ID3_with_depth_restriction(trainingFolds, attributes, 0, k)

        # test tree against withheld set
        count = 0
        correct = 0
        for test in testFold:
            label = test[0]
            result = test_tree(root, test)
            if result == label:
                correct += 1
            count += 1

        acccuracies.append(correct / count)
    # mean
    mean = sum(acccuracies) / len(acccuracies)
    # variance
    variance = 0
    for accuracy in acccuracies:
        variance += ((accuracy - mean)**2)
    variance = variance / len(acccuracies)
    # standard deviation
    standardDeviation = sqrt(variance)
    print('Depth: ' + str(k) + '\tAverage Accuracy: ' + str(mean) + 
        '\tStandard Deviation: ' + str(standardDeviation)) ### PRINTf ###

    if (mean > greatestAccuracy):
        greatestAccuracy = mean
        bestDepth = k
print('\nBest Depth: ' + str(bestDepth) + '\n') ### PRINTg ###

# retry original data w/ optimal depth
attributes = get_measured_attributes(examples)
root = ID3_with_depth_restriction(examples, attributes, 0, bestDepth)

count = 0
correct = 0
for example in examples:
    label = example[0]  # store for validation
    result = test_tree(root, example)
    if result == label:
        correct += 1
    count += 1

#print('Training Set Accuracy w/ Best Depth: ' + str(correct / count) + '\n')

# test data
tests = read_file('data/a1a.test')

count = 0
correct = 0
for test in tests:
    label = test[0]
    result = test_tree(root, test)
    if result == label:
        correct += 1
    count += 1

print('Test Set Accuracy w/ Best Depth: ' + str(correct / count) + '\n') ### PRINTh ###