class Node:

    def __init__(self, value):
        self.value = value
        self.children = {}

    def add_child(self, branch, node):
        self.children[branch] = node

    def get_depth(self):
        if len(self.children) == 0:
            return 0
        else:
            lDepth = []
            for child in self.children:
                lDepth.append(self.children[child].get_depth())
            return max(lDepth) + 1