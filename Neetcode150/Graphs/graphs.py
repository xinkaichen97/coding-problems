"""
Problems for Graphs
"""
from typing import List


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
