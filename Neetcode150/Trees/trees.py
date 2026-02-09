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


    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        """
        https://neetcode.io/problems/same-binary-tree
        Time: O(n), Space: O(n)
        """
        # edge case: equal if both are None
        if not p and not q:
            return True
        # if both exist and have equal values, check left and right subtrees
        if p and q and p.val == q.val:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        else:
            return False
            

    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        """
        https://neetcode.io/problems/binary-tree-right-side-view
        Time: O(n), Space: O(n)
        """
        # since we won't add None in the queue, check it here
        if not root:
            return []

        res = []
        q = deque([root])

        # BFS
        while q:
            # loop through all items in the queue for the entire level
            for i in range(len(q)):
                node = q.popleft()
                # add node only when it's not None
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)

            # add the last node which is the right side view
            res.append(node.val)
        
        return res

    
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


class Codec:
    """
    https://neetcode.io/problems/serialize-and-deserialize-binary-tree
    Time: O(n), Space: O(n)
    """

    # Encodes a tree to a single string.
    def serialize(self, root: Optional[TreeNode]) -> str:
        res = []
        def dfs(node):
            # if current node is None, add N and return
            if not node:
                res.append("N")
                return
            # otherwise add node.val
            # preorder traversal
            res.append(str(node.val))
            # go to left and right subtrees
            dfs(node.left)
            dfs(node.right)

        dfs(root)
        return ",".join(res)
        
    # Decodes your encoded data to tree.
    def deserialize(self, data: str) -> Optional[TreeNode]:
        vals = data.split(",")
        idx = 0

        def dfs():
            nonlocal idx
            # if current node is None, return None and increment index (go to the next node)
            if vals[idx] == "N":
                idx += 1
                return None
            # otherwise create a new node
            # same order as serialize
            node = TreeNode(int(vals[idx]))
            idx += 1
            # build left and right subtrees
            node.left = dfs()
            node.right = dfs()
            return node

        # run dfs and return
        return dfs()

    # BFS version
    def serialize_bfs(self, root: Optional[TreeNode]) -> str:
        if not root:
            return None
        res = []
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if node:
                res.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                res.append("N")

        return ",".join(res)

    def deserialize(self, data: str) -> Optional[TreeNode]:
        vals = data.split(",")
        if vals[0] == "N":
            return None
        
        root = TreeNode(int(vals[0]))
        queue = deque([root])
        idx = 1
        while queue:
            node = queue.popleft()
            if vals[idx] != "N":
                node.left = TreeNode(int(vals[idx]))
                queue.append(node.left)
            idx += 1
            if vals[idx] != "N":
                node.right = TreeNode(int(vals[idx]))
                queue.append(node.right)
            idx += 1
        return root
                
