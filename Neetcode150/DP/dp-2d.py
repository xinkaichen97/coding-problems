"""
Problems for Dynamic Programming
"""
from typing import List


class Solution:
  
    def uniquePaths(self, m: int, n: int) -> int:
        """
        https://neetcode.io/problems/count-paths
        Time: O(m * n), Space: O(m * n) - can be optimized to O(n)
        """
        # dp[i][j] - the number of unique paths to index i,j
        # initialize as 1 for the first row and column
        dp = [[1] * n for _ in range(m)]
        
        for i in range(1, m):
            for j in range(1, n):
                # just the left plus the top
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

        # # space-optimized version with 1d array: O(n)
        # # dp holds the value of the current row (starting with 1's)
        # dp = [1] * n
        # for i in range(1, m):
        #     for j in range(1, n):
        #         # for every i, we add above; for every j, we add left
        #         dp[j] += dp[j - 1]
        # return dp[n - 1]
        
        return dp[m - 1][n - 1]
