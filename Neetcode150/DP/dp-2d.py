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
      

    def maxProfit(self, prices: List[int]) -> int:
        """
        https://neetcode.io/problems/buy-and-sell-crypto-with-cooldown
        Time: O(n), Space: O(n)
        """
        # top-down DP with memoization
        dp = {}

        def dfs(i, canBuy):
            # base case
            if i >= len(prices):
                return 0
            if (i, canBuy) in dp:
                return dp[(i, canBuy)]

            # always calculate the cooldown value - skip the current day
            cooldown = dfs(i + 1, canBuy)
            # if able to buy
            if canBuy:
                # calculate the profit of buying and compare with cooldown
                res = dfs(i + 1, False) - prices[i]
                dp[(i, canBuy)] = max(res, cooldown)
            else:
                # calculate the profit of selling and compare with cooldown
                res = dfs(i + 2, True) + prices[i] # need to skip the next day
                dp[(i, canBuy)] = max(res, cooldown)
            return dp[(i, canBuy)]
        
        return dfs(0, True)

        # # bottom-up DP
        # n = len(prices)
        # # dp[i][0] - max profit from day i if can sell, dp[i][1] - if can buy
        # dp = [[0, 0] for _ in range(n + 1)]
        # for i in range(n - 1, -1, -1):
        #     for buying in [True, False]:
        #         if buying:
        #             # buy - profit if buying today; cooldown - profit if not buying
        #             buy = dp[i + 1][0] - prices[i] if i + 1 < n else -prices[i] # if out of bounds, dp value = 0
        #             cooldown = dp[i + 1][1] if i + 1 < n else 0
        #             dp[i][1] = max(buy, cooldown)
        #         else:
        #             # sell - profit if selling today; can buy on day i + 2
        #             sell = dp[i + 2][1] + prices[i] if i + 2 < n else prices[i]
        #             cooldown = dp[i + 1][0] if i + 1 < n else 0
        #             dp[i][0] = max(sell, cooldown)
        # return dp[0][1]


    def change(self, amount: int, coins: List[int]) -> int:
        """
        https://neetcode.io/problems/coin-change-ii
        Time: O(n * a), Space: O(n * a)
        """
        # top-down DP with memoization
        dp = {}
        def dfs(i, cursum):
            # base cases
            if cursum == amount:
                return 1
            if i == len(coins) or cursum > amount:
                return 0
            if (i, cursum) in dp:
                return dp[(i, cursum)]

            # either pick the current coin (keep the current index for re-picking) or skip
            dp[(i, cursum)] = dfs(i, cursum + coins[i]) + dfs(i + 1, cursum)
            return dp[(i, cursum)]
        
        return dfs(0, 0)

        # # bottom-up DP
        # n = len(coins)
        # # dp[i][a]: the number of ways to form amount a using coins from index i onward
        # dp = [[0] * (amount + 1) for _ in range(n + 1)]  # (n+1) * (amount+1)
        # for i in range(n + 1):
        #     dp[i][0] = 1  # initialize the first column as 1 - one way to get amount 0 (no coin)
        # for i in range(n - 1, -1, -1):  
        #     for a in range(amount + 1):
        #         dp[i][a] = dp[i + 1][a]  # always copy from i+1
        #         if a >= coins[i]: 
        #             dp[i][a] += dp[i][a - coins[i]]
        # return dp[0][amount]

        # # O(a) Space solution
        # dp = [0] * (amount + 1)
        # dp[0] = 1
        # for i in range(len(coins) - 1, -1, -1):
        #     for a in range(1, amount + 1):
        #         dp[a] += dp[a - coins[i]] if coins[i] <= a else 0
        # return dp[amount]


    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        """
        https://neetcode.io/problems/target-sum
        Time: O(n * m), Space: O(n * m), n - len(nums), m - sum(nums)
        """
        # top-down DP with memoization
        memo = {}
        def dfs(i, cursum):
            # when reaching the end, check if cursum is the same as target
            if i == len(nums):
                return cursum == target
            if (i, cursum) in memo:
                return memo[(i, cursum)]
            # add or subtract nums[i] and go to i + 1
            memo[(i, cursum)] = dfs(i + 1, cursum + nums[i]) + dfs(i + 1, cursum - nums[i])
            return memo[(i, cursum)]
          
        return dfs(0, 0)
      
        # # bottom-up DP
        # n = len(nums)
        # # dp[i] - the counts for each sum using the first i numbers
        # dp = [defaultdict(int) for _ in range(n + 1)]  # cannot do [defaultdict(int)] * (n + 1)
        # dp[0][0] = 1
        # for i in range(n):
        #     for total, count in dp[i].items():  # add counts to dp[i + 1]
        #         dp[i + 1][total + nums[i]] += count
        #         dp[i + 1][total - nums[i]] += count
        # return dp[n][target]

        # # O(m) Space solution
        # dp = defaultdict(int)
        # dp[0] = 1
        # for i in range(n):
        #     nextdp = defaultdict(int)
        #     for total, count in dp.items():
        #         nextdp[total + nums[i]] += count
        #         nextdp[total - nums[i]] += count
        #     dp = nextdp
        # return dp[target]

  
    def minDistance(self, word1: str, word2: str) -> int:
        """
        https://neetcode.io/problems/edit-distance
        Time: O(m * n), Space: O(m * n)
        """
        len1, len2 = len(word1), len(word2)
        # base case
        if len1 == 0:
            return len2
        if len2 == 0:
            return len1

        # create dp matrix of (m+1) * (n+1)
        # dp[i][j] - the edit distance between word1[i:] and word2[j:]
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        # update the base cases (empty strings at the bottom)
        for i in range(len1 + 1):
            dp[i][len2] = len1 - i
        for j in range(len2 + 1):
            dp[len1][j] = len2 - j

        # bottom-up
        for i in range(len1 - 1, -1, -1):
            for j in range(len2 - 1, -1, -1):
                # if the current characters are the same, no need to edit at this position
                if word1[i] == word2[j]:
                    dp[i][j] = dp[i + 1][j + 1]
                # otherwise, take the minimum of insert (j + 1), delete (i + 1), or replace, and plus 1
                else:
                    dp[i][j] = 1 + min(dp[i][j + 1], dp[i + 1][j], dp[i + 1][j + 1])
        
        return dp[0][0]
      
