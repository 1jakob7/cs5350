from math import log2
from node import Node

def getOverallEntropy(examples):
    pos = 0
    neg = 0
    tot = len(examples)
    for example in examples:
        if example[0] == '1':
            pos += 1
        else:
            neg += 1
    return (-(pos/tot)*log2(pos/tot)) - ((neg/tot)*log2(neg/tot))

def getAttributeEntropy(examples, attribute):
    posLabelCount = 0
    negLabelCount = 0
    posAttributeCount = 0
    negAttributeCount = 0

    for example in examples:
        if (attribute + ':1') in example:
            if example[0] == '1':
                posLabelCount += 1
            posAttributeCount += 1
        else:
            if example[0] == '1':
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

def findHighestInfoGain(examples, attributes):
    currentEntropy = getOverallEntropy(examples)
    highestInfoGain = 0
    bestAttribute = attributes[0]
    for attribute in attributes:
        aEntropy = getAttributeEntropy(examples, attribute)
        aInfoGain = currentEntropy - aEntropy
        if aInfoGain > highestInfoGain:
            highestInfoGain = aInfoGain
            bestAttribute = attribute

    return bestAttribute

def getExamplesWithAttributeValue(examples, attributeValue):
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

def getCommonLabel(examples):
    pos = 0
    for example in examples:
        if example[0] == int(1):
            pos += 1
    if pos > (len(examples) - pos):
        return '1' # was +1
    else:
        return '0' # was -1

def getAttributeValues(attribute):
    return ['0', '1']

def depthRestrictedID3(data, attributes, currentDepth, maxDepth):
    # if all examples have the same label -> return node tree w/ that label
    lFlag = True
    l = data[0][0]
    for example in data:
        if l != example[0]:
            lFlag = False
            break
    if lFlag:
        return Node(l)

    # get most common label if no attributes left or we've reach the max depth for the tree
    if len(attributes) < 1 or currentDepth == maxDepth:
        return Node(getCommonLabel(data))

    # alright then, we're recursing
    bestAttribute = findHighestInfoGain(data, attributes)
    root = Node(bestAttribute)

    attributesSubset = attributes.copy()
    attributesSubset.remove(bestAttribute)
    # half-assed attempt to include non-binary functionality
    for value in getAttributeValues(bestAttribute):
        attributeValue = bestAttribute + ':' + value
        dataSubset = getExamplesWithAttributeValue(data, attributeValue)
        if len(dataSubset) == 0:
            root.add_child(value, Node(getCommonLabel(dataSubset)))
        else:
            currentDepth += 1
            root.add_child(value, depthRestrictedID3(dataSubset, attributesSubset, currentDepth, maxDepth))
            currentDepth -= 1 # reset current depth for next child node

    return root