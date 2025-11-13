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
  
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        q = collections.deque()

        # add to queue if not None
        if root:
            q.append(root)

        while q:
            # iterate over all nodes in the current level
            for i in range(len(q)):
                node = q.popleft()
                # add left and right if not None
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
                    
            # add the last node value to res
            res.append(node.val)

        return res


    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        # initialize 
        max_sum, res, level = float('-inf'), 0, 0

        # deque for BFS
        q = collections.deque()
        q.append(root)

        while q:
            # level starts with 1
            level += 1
            curr_sum = 0
          
            # iterate over all the nodes in the current level
            for _ in range(len(q)):
                node = q.popleft()
                curr_sum += node.val
              
                # add left and right if not None
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)

            # update max and level
            if max_sum < curr_sum:
                max_sum, res = curr_sum, level
           
        return res
      
