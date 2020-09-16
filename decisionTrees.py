from math import log2
from node import Node

def read_file():
    data = []
    # open file for reading, "with" handles automatically closing the file
    with open('data/a1a.train', 'r') as file:
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

#####

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
                negLabelCount += 1 # misnomer
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

def get_common_label(examples):
    pos = 0
    for example in examples:
        if example[0] == int(1):
            pos += 1
    if pos > (len(examples) - pos):
        return 1
    else:
        return 0


def ID3(examples, attributes):
    # if all examples have the same label -> return node tree w/ that label
    lFlag = True
    l = examples[0][0]
    for example in examples:
        if l != example[0]:
            lFlag = False
            break
    if lFlag:
        return Node(int(l))

    # create root node
    # ...

    if len(attributes) < 1:
        return Node(get_common_label(examples))

    bestAttribute = find_highest_info_gain(examples, attributes)

    # create root node now?
    root = Node(bestAttribute)

    attributesSubset = attributes
    attributesSubset.remove(bestAttribute)
    # half-assed attempt to include non-binary functionality
    for value in get_attribute_values(bestAttribute):
        #root.add_branch(value)
        attributeValue = bestAttribute + ':' + value
        examplesSubset = get_examples_with_attribute_value(examples, attributeValue)
        if len(examplesSubset) == 0:
            child = Node(get_common_label(examplesSubset))
        else:
            child = Node(ID3(examplesSubset, attributesSubset))
        root.add_child(value, child)

    return root


def test_tree(root, example):
    while len(root.children) > 0:
        if (root.value + ':1') in example:
            root = root.children['1']
        else:
            root = root.children['0']
    return root.value

# *something to keep in mind: only leaf nodes will hold label values (1 or 0), unless referring to index 1 or 0...*

# Main program
examples = read_file()
attributes = get_measured_attributes(examples)

root = ID3(examples, attributes)

# test tree on training set (previous examples)
count = 0
correct = 0
for example in examples:
    label = example.pop(0) # remove label and store for validation
    result = test_tree(root, example)
    if result == label:
        correct += 1
    count += 1 

print('Total: ' + str(count) + '\tCorrect: ' + str(correct) + '\tAccuracy: ' + str(correct / count))

# root.print_tree()
