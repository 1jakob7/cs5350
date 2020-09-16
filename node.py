# created using reference from: www.educative.io/edpresso/binary-trees-in-python

class Node:

    def __init__(self, value):
        self.value = value
        self.children = {}

    # def add_branch(self, branch):
    #     self.children[branch] = None

    def add_child(self, branch, node):
        self.children[branch] = node
