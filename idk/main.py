import csv
import random
import numpy as np

def readFile(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            example = line.split()
            data.append(example)
    return data

def extractLabels(data):
    labels = []
    for example in data:
        labels.append(example[0])
    return labels

def readMiscFile(path):
    data = []
    with open(path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        first = True
        for row in csv_reader:
            if not first:
                for i in range(len(row)): # might have to nix, split instead
                    row[i] = str(i+1) + ':' + str(row[i])
                data.append(row)
            else:
                first = False
    return data

def createVectorsFromMiscs(data, labels):
    attr = {}
    for example in data:
        for attribute in example:
            spl = attribute.split(':')
            attr.setdefault(spl[0], set()).add(spl[1])
    categories = []
    for key in attr.keys():
        categories.append(list(attr[key]))
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = categories[j].index(data[i][j].split(':')[1])
        data[i][:0] = [int(labels[i])]
    return data


def getNumberOfAttributes(data):
    hmmm = 0

# Main
# setup global constants...
basePath = 'project_data/data/'

# read initial data
trainData = readFile(basePath + 'glove/glove.train.libsvm')
trainMiscData = readMiscFile(basePath + 'misc-attributes/misc-attributes-train.csv')
# ...

labels = extractLabels(trainData)
trainMiscData = createVectorsFromMiscs(trainMiscData, labels)

folds = []