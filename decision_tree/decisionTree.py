from math import log2
from node import Node

def DepthRestrictedID3(data, attributes, currentDepth, maxDepth):
    # if all examples have the same label -> return node tree w/ that label
    lFlag = True
    l = data[0][0]
    for example in data:
        if l != example[0]:
            lFlag = False
            break
    if lFlag:
        return Node(l)