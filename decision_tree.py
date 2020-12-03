from math import log2
from node import Node

def getOverallEntropy(examples):
    pos = 0
    neg = 0
    tot = len(examples)
    for example in examples:
        if example[0] == 1:
            pos += 1
        else:
            neg += 1
    return (-(pos/tot)*(log2(pos/tot))) - ((neg/tot)*(log2(neg/tot)))

def getAttributeEntropy(examples, attribute):
    posLabelCount = 0
    negLabelCount = 0
    posAttributeCount = 0
    negAttributeCount = 0

    for example in examples:
        if example[attribute] == 1:
            if example[0] == 1:
                posLabelCount += 1
            posAttributeCount += 1
        else:
            if example[0] == 1:
                negLabelCount += 1  # misnomer
            negAttributeCount += 1

    pos = posLabelCount
    neg = posAttributeCount - posLabelCount
    tot = posAttributeCount
    posEntropy = 0
    if (pos != 0 and neg != 0):
        posEntropy = (-(pos/tot)*(log2(pos/tot))) - ((neg/tot)*(log2(neg/tot)))
    pos = negLabelCount
    neg = negAttributeCount - negLabelCount
    tot = negAttributeCount
    negEntropy = 0
    if (pos != 0 and neg != 0):
        negEntropy = (-(pos/tot)*(log2(pos/tot))) - ((neg/tot)*(log2(neg/tot)))

    return ((posAttributeCount / len(examples))*posEntropy) + ((negAttributeCount / len(examples))*negEntropy)

def findHighestInfoGain(examples, attributes):
    currentEntropy = getOverallEntropy(examples)
    highestInfoGain = 0
    bestAttribute = attributes[0]
    for attribute in attributes:
        attrEntropy = getAttributeEntropy(examples, attribute)
        attrInfoGain = currentEntropy - attrEntropy
        if attrInfoGain > highestInfoGain:
            highestInfoGain = attrInfoGain
            bestAttribute = attribute
    return bestAttribute

def getExamplesWithAttributeValue(examples, attribute, attrValue):
    subset = []
    for example in examples:
        if example[attribute] == attrValue:
            subset.append(example)
    return subset

def getCommonLabel(examples):
    pos = 0
    for example in examples:
        if example[0] == 1:
            pos += 1
    if pos > (len(examples) - pos):
        return 1
    else:
        return -1

def getAttributeValues(attribute):
    return [-1, 1]

def ID3(data, attributes, currentDepth, maxDepth):
    # if all examples have the same label -> return node tree w/ that label
    lFlag = True
    l = data[0][0]
    for example in data:
        if l != example[0]:
            lFlag = False
            break
    if lFlag:
        return Node(l)

    # get most common label if no attributes are left or we've reach the max depth for the tree
    if len(attributes) < 1 or currentDepth == maxDepth:
        return Node(getCommonLabel(data))

    # alright then, we're recursing
    bestAttribute = findHighestInfoGain(data, attributes)
    root = Node(bestAttribute)

    attributeSubset = attributes.copy()
    attributeSubset.remove(bestAttribute)
    # half-assed attempt to include non-binary functionality
    for value in getAttributeValues(bestAttribute):
        dataSubset = getExamplesWithAttributeValue(data, bestAttribute, value)
        if len(dataSubset) == 0:
            root.add_child(value, Node(getCommonLabel(data)))
        else:
            currentDepth += 1
            root.add_child(value, ID3(dataSubset, attributeSubset, currentDepth, maxDepth))
            currentDepth -= 1 # reset current depth for next child node

    return root