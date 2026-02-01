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
        # dp[i][j] - the number of unique paths to index (i, j)
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


    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        """
        https://neetcode.io/problems/count-paths
        Time: O(m * n), Space: O(m * n) - can be optimized to O(n)
        """
        # dp[i][j] - LCS from index (i, j)
        dp = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]

        # go from bottom-right to top-left
        for i in range(len(text1) - 1, -1, -1):
            for j in range(len(text2) - 1, -1, -1):
                # if the character is the same, the result is LCS(i+1, j+1) + 1
                if text1[i] == text2[j]:
                    dp[i][j] = 1 + dp[i + 1][j + 1]
                else:
                    # if not, the result is the max of advancing one char in either text
                    dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
                    
        return dp[0][0]
      
