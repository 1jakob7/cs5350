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

def separate_labels_from_features(examples):
    labels = []
    for example in examples:
        labels.append(example.pop(0))
    return labels

def get_positive_attribute_count(attributes, a):
    pos = 0
    for aList in attributes:
        for attribute in aList:
            if attribute == a:
                pos += 1
    return pos
    
def find_highest_info_gain(labels, attributes):
    pos = 0
    neg = 0
    tot = len(labels)

    for label in labels:
        if int(label) > 0:
            pos += 1
        else:
            neg += 1
    currentEntropy = -(pos/tot)*log2(pos/tot) - (neg/tot)*log2(neg/tot)

    foundIndexes = []
    highestInfoGain = 0
    
    for aList in attributes:
        if (len(aList) > 0):
            bestAttribute = aList[0].split(':').pop(0)

    for aList in attributes:
        for attribute in aList:
            aIndex = attribute.split(':').pop(0)
            aValue = attribute.split(':').pop(1)
            if aIndex not in foundIndexes:
                foundIndexes.append(aIndex)
                pos = get_positive_attribute_count(attributes, attribute)
                neg = tot - pos
                if pos == 0 or neg == 0:
                    aEntropy = 0
                else:
                    aEntropy = -(pos/tot)*log2(pos/tot) - (neg/tot)*log2(neg/tot)
                aInfoGain = currentEntropy - aEntropy
                if aInfoGain > highestInfoGain:
                    highestInfoGain = aInfoGain
                    bestAttribute = aIndex
                    
    return bestAttribute

# dios mio!
def fill_attribute_examples(labels, attributes, bestAttribute, posLabels, posAttributes, negLabels, negAttributes):
    for i in range(len(attributes)):
        if bestAttribute in attributes[i]:
            posLabels.append(labels[i])
            posAttributes.append(attributes[i])
        else:
            negLabels.append(labels[i])
            negAttributes.append(attributes[i])

def remove_attribute(attributes, a):
    for aList in attributes:
        aList.remove(a)
    return attributes

def get_common_label(labels):
    pos = 0
    neg = 0
    for label in labels:
        if label == 1:
            pos += 1
        else:
            neg += 1
    if pos > neg:
        return 1
    else:
        return 0

def ID3(labels, attributes):
    # if all examples have the same label -> return node tree w/ that label
    lFlag = True
    l = labels[0]
    for label in labels:
        if label != l:
            lFlag = False
            break
    if lFlag:
        return Node(l)

    # create root node
    # ...

    # get attribute that best classifies the set of examples
    bestAttribute = find_highest_info_gain(labels, attributes)

    # create root node now?
    root = Node(bestAttribute)

    posLabels = []
    posAttributes = []
    negLabels = []
    negAttributes = []
    fill_attribute_examples(labels, attributes, bestAttribute + ':1', posLabels, posAttributes, negLabels, negAttributes)

    if len(posAttributes) == 0:
        # insert right(1) leaf node w/ end value 0 or 1 
        root.insert(1, Node(get_common_label(posLabels)))
    else:
        posAttributes = remove_attribute(posAttributes, bestAttribute + ':1')
        rNode = ID3(posLabels, posAttributes)
        root.insert(1, rNode)

    if len(negAttributes) == 0:
        # insert left(0) leaf node w/ end value 0 or 1
        root.insert(0, Node(get_common_label(negLabels)))
    else:
        lNode = ID3(negLabels, negAttributes)
        root.insert(0, lNode)

    return root


# *something to keep in mind: only leaf nodes will hold label values (1 or 0), unless referring to index 1 or 0...*

# Main program
attributes = read_file() # slight misnomer, since labels are still present...
labels = separate_labels_from_features(attributes) # ah, that's better

root = ID3(labels, attributes)

root.print_tree()

#print(str(labels[1]) + " --- " + str(attributes[1]))

