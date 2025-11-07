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
      

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """
        https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/
        Time: O(N), Space: O(H)
        """
        def dfs(root):
            # base case: if root is None, return None
            # if p or q is the root, return the root
            if not root or root == p or root == q:
                return root
            # check left and right
            left = dfs(root.left)
            right = dfs(root.right)
            # if both left and right are not None, current root is the LCA
            if left and right:
                return root
            # if not, return left or right
            return left or right
            
        return dfs(root)


    def goodNodes(self, root: TreeNode) -> int:
        """
        https://leetcode.com/problems/count-good-nodes-in-binary-tree/
        Time: O(N), Space: O(H)
        """
        def dfs(root, val):
            # base case if root is None
            if not root:
                return 0
            # if the current node value is greater
            # add one to the count and update the max val
            if root.val >= val:
                return 1 + dfs(root.left, root.val) + dfs(root.right, root.val)
            # if not, current node is not a good node, do not increment the count
            # but still need to check its left and right, while keeping the max val
            else:
                return dfs(root.left, val) + dfs(root.right, val)
        return dfs(root, root.val)
        

    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        """
        https://leetcode.com/problems/path-sum-iii/
        Time: O(N), Space: O(H + N)
        """
        # prefix dict to store count of cumulative sums
        # initiate prefix dict with 0
        prefix = {0: 1}

        def dfs(node, currSum):
            if not node:
                return 0

            # update current sum
            currSum += node.val
            # see if any previous node satisfies target
            count = prefix.get(currSum - targetSum, 0)
            # update prefix with current sum
            prefix[currSum] = prefix.get(currSum, 0) + 1
            
            # look into left and right
            count += dfs(node.left, currSum)
            count += dfs(node.right, currSum)

            # backtrack: decrement current sum from prefix after recursion
            prefix[currSum] -= 1
            
            return count

        return dfs(root, 0)
        
