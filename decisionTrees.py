from os import listdir
from math import log2
from math import sqrt
from node import Node

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

def get_attribute_entropy(examples, attribute):
    posLabelCount = 0
    negLabelCount = 0
    posAttributeCount = 0
    negAttributeCount = 0

    for example in examples:
        if (attribute + ':1') in example:
            if example[0] == '+1':
                posLabelCount += 1
            posAttributeCount += 1
        else:
            if example[0] == '+1':
                negLabelCount += 1  # misnomer
            negAttributeCount += 1

    pos = posLabelCount
    neg = posAttributeCount - posLabelCount
    tot = posAttributeCount
    posEntropy = 0
    if (pos != 0 and neg != 0):
        posEntropy = (-(pos/tot)*log2(pos/tot)) - ((neg/tot)*log2(neg/tot))
    pos = negLabelCount
    neg = negAttributeCount - negLabelCount
    tot = negAttributeCount
    negEntropy = 0
    if (pos != 0 and neg != 0):
        negEntropy = (-(pos/tot)*log2(pos/tot)) - ((neg/tot)*log2(neg/tot))

    return ((posLabelCount / len(examples))*(posEntropy) + (negLabelCount / len(examples))*(negEntropy))


def find_highest_info_gain(examples, attributes):
    pos = 0
    neg = 0
    tot = len(examples)

    for example in examples:
        if example[0] == '+1':
            pos += 1
        else:
            neg += 1
    currentEntropy = (-(pos/tot)*log2(pos/tot)) - ((neg/tot)*log2(neg/tot))

    highestInfoGain = 0
    bestAttribute = attributes[0]
    for attribute in attributes:
        aEntropy = get_attribute_entropy(examples, attribute)
        aInfoGain = currentEntropy - aEntropy
        if aInfoGain > highestInfoGain:
            highestInfoGain = aInfoGain
            bestAttribute = attribute

    return bestAttribute

# extension of the bs
def get_attribute_values(attribute):
    return ['0', '1']

def get_common_label(examples):
    pos = 0
    for example in examples:
        if example[0] == int(1):
            pos += 1
    if pos > (len(examples) - pos):
        return '+1'
    else:
        return '-1'

def get_examples_with_attribute_value(examples, attributeValue):
    value = attributeValue.split(':').pop(1)
    subset = []
    if value == '0':
        attributeValue = attributeValue.split(':').pop(0) + ':1'
        for example in examples:
            if attributeValue not in example:
                subset.append(example)
    else:
        for example in examples:
            if attributeValue in example:
                subset.append(example)
    return subset

def ID3(examples, attributes ):
    # if all examples have the same label -> return node tree w/ that label
    lFlag = True
    l = examples[0][0]
    for example in examples:
        if l != example[0]:
            lFlag = False
            break
    if lFlag:
        return Node(l)

    # get most common label if no attributes left
    if len(attributes) < 1:
        return Node(get_common_label(examples))

    # alright then, we're recursing
    bestAttribute = find_highest_info_gain(examples, attributes)
    root = Node(bestAttribute)

    attributesSubset = attributes
    attributesSubset.remove(bestAttribute)
    # half-assed attempt to include non-binary functionality
    for value in get_attribute_values(bestAttribute):
        attributeValue = bestAttribute + ':' + value
        examplesSubset = get_examples_with_attribute_value(
            examples, attributeValue)
        if len(examplesSubset) == 0:
            root.add_child(value, Node(get_common_label(examplesSubset)))
        else:
            root.add_child(value, ID3(examplesSubset, attributesSubset))

    return root

def ID3_with_depth_restriction(examples, attributes, currentDepth, maxDepth):
    # if all examples have the same label -> return node tree w/ that label
    lFlag = True
    l = examples[0][0]
    for example in examples:
        if l != example[0]:
            lFlag = False
            break
    if lFlag:
        return Node(l)

    # get most common label if no attributes left or we've reach the max depth for the tree
    if len(attributes) < 1 or currentDepth == maxDepth:
        return Node(get_common_label(examples))

    # alright then, we're recursing
    bestAttribute = find_highest_info_gain(examples, attributes)
    root = Node(bestAttribute)

    attributesSubset = attributes
    attributesSubset.remove(bestAttribute)
    # half-assed attempt to include non-binary functionality
    for value in get_attribute_values(bestAttribute):
        attributeValue = bestAttribute + ':' + value
        examplesSubset = get_examples_with_attribute_value(
            examples, attributeValue)
        if len(examplesSubset) == 0:
            root.add_child(value, Node(get_common_label(examplesSubset)))
        else:
            currentDepth += 1
            root.add_child(value, ID3_with_depth_restriction(examplesSubset, attributesSubset, currentDepth, maxDepth))
            currentDepth -= 1 # reset current depth for next child node

    return root

def test_tree(root, example):
    while len(root.children) > 0:
        if (root.value + ':1') in example:
            root = root.children['1']
        else:
            root = root.children['0']
    return root.value

# Main program
# setup and training data
examples = read_file('data/a1a.train')
attributes = get_measured_attributes(examples)
root = ID3(examples, attributes)

count = 0
correct = 0
for example in examples:
    label = example[0]  # store for validation
    result = test_tree(root, example)
    if result == label:
        correct += 1
    count += 1

print('Training Data: Total: ' + str(count) + '\tCorrect: ' +
      str(correct) + '\tAccuracy: ' + str(correct / count))

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

print('Testing Data: Total: ' + str(count) + '\tCorrect: ' +
    str(correct) + '\tAccuracy: ' + str(correct / count) + '\n')

# limiting depth w/ 5-fold cross-validation
dir = 'data/CVfolds/'
folds = {}
count = 0
for fileName in listdir(dir):
    foldData = read_file(dir + fileName)
    folds[count] = foldData
    count += 1

for k in range(41):
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

    print('Depth: ' + str(k) + '\tMean: ' + str(mean) + '\tStandard Dev: ' + str(standardDeviation))


# print('Fold Data ' + str(k) + ': Total: ' + str(count) + '\tCorrect: ' +
#             str(correct) + '\tAccuracy: ' + str(correct / count))