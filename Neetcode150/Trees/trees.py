"""
Problems for Trees
"""
from collections import deque
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


    def maxDepth(self, root: Optional[TreeNode]) -> int:
        """
        https://neetcode.io/problems/depth-of-binary-tree
        Time: O(n), Space: O(h)
        """
        if not root:
            return 0

        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))


    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        """
        https://neetcode.io/problems/binary-tree-maximum-path-sum
        Time: O(n), Space: O(n)
        """
        # initialize res with root.val or -infinity
        res = root.val

        def dfs(root):
            # use nonlocal to access res, or define res as a list (mutable)
            nonlocal res
            # base case: return 0 if None
            if not root:
                return 0
                
            # get max of left and right subtrees
            # if negative, return 0 to not include the subtree
            leftMax = max(0, dfs(root.left))
            rightMax = max(0, dfs(root.right))

            # update res with root.val plus both left and right
            res = max(res, root.val + leftMax + rightMax)
            # but only return the max of left OR right in the dfs function
            return root.val + max(leftMax, rightMax)

        dfs(root)
        return res
        
