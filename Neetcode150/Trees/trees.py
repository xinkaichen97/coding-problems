"""
Problems for Trees
"""
from typing import Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
  
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        https://neetcode.io/problems/invert-a-binary-tree
        Time: O(n), Space: O(n)
        """
        # edge case - root is None
        if not root: 
            return None

        # Solution 1: recursive DFS
        # swap left and right nodes
        root.left, root.right = root.right, root.left

        # recursively reverse left and right subtrees
        self.invertTree(root.left)
        self.invertTree(root.right)

        # # Solution 2: iterative DFS using a stack
        # stack = [root]
        # while stack:
        #     node = stack.pop()
        #     node.left, node.right = node.right, node.left
        #     if node.left:
        #         stack.append(node.left)
        #     if node.right:
        #         stack.append(node.right)

        # # Solution 3: BFS using a queue
        # queue = deque([root])
        # while queue:
        #     node = queue.popleft()
        #     node.left, node.right = node.right, node.left
        #     if node.left:
        #         queue.append(node.left)
        #     if node.right:
        #         queue.append(node.right)
              
        return root

  
