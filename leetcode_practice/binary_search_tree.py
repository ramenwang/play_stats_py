class Node:
    def __init__(self, info):
        self.info = info  
        self.left = None  
        self.right = None 
        self.level = None 

    def __str__(self):
        return str(self.info) 

def preOrder(root):
    if root == None:
        return
    print (root.info, end=" ")
    preOrder(root.left)
    preOrder(root.right)
    
class BinarySearchTree:
    def __init__(self): 
        self.root = None

#Node is defined as
#self.left (the left child of the node)
#self.right (the right child of the node)
#self.info (the value of the node)

    def insert(self, val):
        
        #Enter you code here.
        def insertion1(pointer, val):
            print("before:", pointer)
            if pointer == None:
                # print("occurred!")
                pointer = Node(val)
            else:
                if val > pointer.info:
                    insertion1(pointer.right, val)
                else:
                    insertion1(pointer.left, val)
            print("after: ",pointer)
            #return pointer

        def insertion2(pointer, val):
            print("before:",pointer)
            if pointer == None:
                pointer = Node(val)
            else:
                if val > pointer.info:
                    if pointer.right == None:
                        pointer.right = Node(val)
                    else: 
                        insertion2(pointer.right, val)
                else:
                    if pointer.left == None:
                        pointer.left = Node(val)
                    else:
                        insertion2(pointer.left, val)
            print("after:",pointer)

        if self.root == None:
            self.root = Node(val)
        else:
            insertion1(self.root, val)   
  
# Driver program to test the above functions 
# Let us create the following BST 
#      50 
#    /      \ 
#   30     70 
#   / \    / \ 
#  20 40  60 80 
r = BinarySearchTree()
numList = [4,2,3,1,7,6]
for i in numList:
    print("input ------ ",i)
    r.insert(i)
    
# Print inoder traversal of the BST 
preOrder(r.root) 