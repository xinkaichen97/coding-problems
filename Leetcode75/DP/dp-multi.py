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
        Time: O(m * n), Space: O(m * n)
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
      

    def maxProfit(self, prices: List[int], fee: int) -> int:
        """
        https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee
        Time: O(n), Space: O(n)
        """
        # use two dp arrays
        # hold[i] - max profit if holding at day i
        # free[i] - max profit if not holding at day i
        n = len(prices)
        hold, free = [0] * n, [0] * n

        # initialize hold with the buy-in price at day 0
        hold[0] = -prices[0]
        
        for i in range(1, n):
            # if hold/buy, it's the max of holding on the previous day, or buying at the current price
            hold[i] = max(hold[i - 1], free[i - 1] - prices[i])
            # if wait/sell, it's the max of freeing on the previous day, or the gain/loss from selling
            free[i] = max(free[i - 1], hold[i - 1] + prices[i] - fee)

        # the result is the last item in free as there's no point holding
        return free[-1]

        # # O(1) space version
        # hold, free = -prices[0], 0
        # for i in range(1, n):
        #     tmp = hold
        #     hold = max(hold, free - prices[i])
        #     free = max(free, tmp + prices[i] - fee)
        # return free
      

    def minDistance(self, word1: str, word2: str) -> int:
        """
        https://leetcode.com/problems/edit-distance
        Time: O(m * n), Space: O(m * n)
        """
        len1, len2 = len(word1), len(word2)
        # edge case: if one string is empty, the edit distance is just the length of the other
        if len1 == 0:
            return len2
        if len2 == 0:
            return len1
          
        # dp[i][j] - the edit distance of word1[:i] and word2[:j] (starting from "")
        dp = [[0 for _ in range(len2 + 1)] for _ in range(len1 + 1)]
        # fill the first row and column: edit distance with the empty string is just its length
        for i in range(1, len1 + 1):
            dp[i][0] = i
        for j in range(1, len2 + 1):
            dp[0][j] = j
          
        # check every combination from index 1
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                # if the character is the same, do nothing
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    # if not, get the min of insert (j - 1), delete (i - 1), and replace (i - 1 & j - 1)
                    # plus 1 for the current operation
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                  
        return dp[len1][len2]
      
