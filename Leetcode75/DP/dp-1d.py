"""
Problems for Dynamic Programming
"""


class Solution:
  
    def tribonacci(self, n: int) -> int:
        """
        https://leetcode.com/problems/n-th-tribonacci-number
        Time: O(N), Space: O(N)
        """
        # base case
        if n == 0:
            return 0
        if n in [1, 2]:
            return 1
        
        # initialize DP array
        dp = [0] * (n + 1)
        dp[1] = 1
        dp[2] = 1

        # start from the beginning
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2] + dp[i - 3]

        # return the n-th value
        return dp[n]


    def minCostClimbingStairs(self, cost: List[int]) -> int:
        """
        https://leetcode.com/problems/min-cost-climbing-stairs
        Time: O(N), Space: O(N)
        """
        # initialize DP array
        n = len(cost)
        dp = [0] * (n + 1)

        # the last value is 0 -> no cost in reaching the end
        # the second last value is the cost of the last stair
        dp[n - 1] = cost[-1]

        # start from the end 
        for i in range(n - 2, -1, -1):
            dp[i] = cost[i] + min(dp[i + 1], dp[i + 2])

        # return the minimum of first and second items
        return min(dp[0], dp[1])
      
