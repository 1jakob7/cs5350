from math import log2
from node import Node

def get_overall_entropy(examples):
    pos = 0
    neg = 0
    tot = len(examples)

    for example in examples:
        if example[0] == '+1':
            pos += 1
        else:
            neg += 1

    return (-(pos/tot)*log2(pos/tot)) - ((neg/tot)*log2(neg/tot))

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
    currentEntropy = get_overall_entropy(examples)
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

def print_highest_info_gain(examples, attributes):
    currentEntropy = get_overall_entropy(examples)
    highestInfoGain = 0
    bestAttribute = attributes[0]
    for attribute in attributes:
        aEntropy = get_attribute_entropy(examples, attribute)
        aInfoGain = currentEntropy - aEntropy
        if aInfoGain > highestInfoGain:
            highestInfoGain = aInfoGain
            bestAttribute = attribute

    print('Best Feature in Training Set: ' + str(bestAttribute) 
        + '\tInformation Gain: ' + str(highestInfoGain) + '\n')

def test_tree(root, example):
    while len(root.children) > 0:
        if (root.value + ':1') in example:
            root = root.children['1']
        else:
            root = root.children['0']

    return root.value