class Node:

    def __init__(self, value):
        self.value = value
        self.children = {}

    def add_child(self, branch, node):
        self.children[branch] = node
