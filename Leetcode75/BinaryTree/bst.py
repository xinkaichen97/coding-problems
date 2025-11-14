"""
Problems for Binary Search Trees
"""
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        """
        https://leetcode.com/problems/search-in-a-binary-search-tree
        Time: O(N), Space: O(H)
        """
        if not root:
            return None
        # return the node if found
        if root.val == val:
            return root
        # search in the left tree if the value is lower
        elif root.val > val:
            return self.searchBST(root.left, val)
        # otherwise search in the right tree
        else:
            return self.searchBST(root.right, val)


    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        """
        https://leetcode.com/problems/delete-node-in-a-bst
        Time: O(H), Space: O(H)
        """
        if not root:
            return None
        # find in the right subtree
        if key < root.val:
            root.left = self.deleteNode(root.left, key)
        # find in the left subtree
        elif key > root.val:
            root.right = self.deleteNode(root.right, key)
        # if val is found
        else:
            # if one subtree is missing, return the other
            if not root.right:
                return root.left
            if not root.left:
                return root.right
              
            # if both left and right children exist
            # look at the right subtree, left subtree is not affected
            temp = root.right
            # find the smallest value in the right subtree to be the successor
            while temp.left:
                temp = temp.left
            # update the value with the successor's value
            root.val = temp.val
            # delete successor from the right subtree
            root.right = self.deleteNode(root.right, temp.val)

            ## alternatively, find the largest value in the left subtree
            # temp = root.left
            # while temp.right:
            #     temp = temp.right
            # root.val = temp.val
            # root.left = self.deleteNode(root.left, temp.val)
          
        return root
      
