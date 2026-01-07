"""
Problems for Dynamic Programming
"""

class Solution:
  
    def uniquePaths(self, m: int, n: int) -> int:
        """
        https://leetcode.com/problems/unique-paths/
        Time: O(m * n), Space: O(m * n)
        """
        ## can be reduced to 1d
        # dp = [1] * n
        # for _ in range(1, m):
        #     for i in range(1, n):
        #         dp[i] += dp[i-1]
        # return dp[-1]
      
        # initialize a 2d array of 1's
        d = [[1] * n for _ in range(m)]

        # loop through each col and each row
        for col in range(1, m):
            for row in range(1, n):
                # the current value is the sum at col - 1 and row - 1
                d[col][row] = d[col - 1][row] + d[col][row - 1]

        # return the last value
        return d[m - 1][n - 1]


    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        """
        https://leetcode.com/problems/longest-common-subsequence
        """
        # initialize a 2d array for each letter in each text
        # the cell values are the LCS from the current point of two texts
        dp = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]

        # loop through each letter
        for col in range(len(text2) - 1, -1, -1):
            for row in range(len(text1) - 1, -1, -1):
                # if the letters are the same, add 1, and both go to the next letter (dp[row+1][col+1])
                if text2[col] == text1[row]:
                    dp[row][col] = 1 + dp[row + 1][col + 1]
                # otherwise, go to the next letter of either text, and take the maximum
                else:
                    dp[row][col] = max(dp[row + 1][col], dp[row][col + 1])

        # return the first value in the table
        return dp[0][0]
      
