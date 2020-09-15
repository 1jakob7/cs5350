# created using reference from: www.educative.io/edpresso/binary-trees-in-python

class Node:

    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
            
    def insert(self, branch, node):
        print ('branch: ' + branch)
        if self.value:
            if branch == 0:
                self.left = node
            else: # == 1
                self.right = node
        else:
            self.value = value
        
    def print_tree(self):
        if self.left:
            self.left.print_tree()
        print(self.value)
        if self.right:
            self.right.print_tree()

#root = Node(17)
#root.insert(0, 5)
#root.insert(1, 12)
#root.print_tree()
