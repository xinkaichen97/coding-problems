"""
Problems for Graphs
"""
from typing import List


# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
      

class Solution:
  
    def numIslands(self, grid: List[List[str]]) -> int:
        """
        https://neetcode.io/problems/count-number-of-islands
        Time: O(m * n), Space: O(m * n)
        """
        # initialize directions
        directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        n_row, n_col = len(grid), len(grid[0])
        res = 0

        # dfs function
        def dfs(r, c):
            # invalid cases: out of bounds
            if r < 0 or c < 0 or r >= n_row or c >= n_col or grid[r][c] == "0":
                return

            # set current cell to 0 to avoid repeated search
            grid[r][c] = "0"
            # run dfs for all four directions
            for dr, dc in directions:
                dfs(r + dr, c + dc)

        # for each cell with 1, run dfs and update res
        for r in range(n_row):
            for c in range(n_col):
                if grid[r][c] == "1":
                    dfs(r, c)
                    res += 1
        
        return res


    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        """
        https://neetcode.io/problems/clone-graph
        Time: O(v + e), Space: O(e)
        """
        # create a mapping from ori to copy
        mapping = {}
        if not node:
            return None

        # dfs function
        def dfs(node):
            # return the copy if already in mapping, so that it doesn't get added more than once
            if node in mapping:
                return mapping[node]

            # create a copy and add to mapping
            copy = Node(node.val)
            mapping[node] = copy
          
            # create copies for all neighbors and add to the copy's neighbors
            for nb in node.neighbors:
                copy.neighbors.append(dfs(nb))
            return copy
        
        return dfs(node)
      
