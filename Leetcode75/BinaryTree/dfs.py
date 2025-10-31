"""
Problems for Binary Trees
"""
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        """
        https://leetcode.com/problems/maximum-depth-of-binary-tree/
        Time: O(N), Space: O(H)
        """
        if not root:
            return 0
        # get the max of left and right subtrees
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))


    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        """
        https://leetcode.com/problems/leaf-similar-trees/
        Time: O(N+M), Space: O(H1+H2)
        """
        # DFS function to return the leaf sequence given a root
        def dfs(root):
            if not root:
                return []
            # if leaf node, return its value
            if not root.left and not root.right:
                return [root.val]
            # check left and right nodes
            return dfs(root.left) + dfs(root.right)
          
        return dfs(root1) == dfs(root2)
      
