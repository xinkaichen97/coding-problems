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

        # dp[n] == 0 -> no cost in reaching the end
        # dp[n - 1] is the cost of the last stair
        dp[n - 1] = cost[-1]

        # start from the end 
        for i in range(n - 2, -1, -1):
            # i-th item is the minimum cost starting from i
            # it's the cost of the i-th stair plus the minimum cost from i+1 or i+2
            dp[i] = cost[i] + min(dp[i + 1], dp[i + 2])

        # return the minimum of first and second items
        return min(dp[0], dp[1])


    def minCostClimbingStairs2(self, cost: List[int]) -> int:
        """
        https://leetcode.com/problems/min-cost-climbing-stairs
        Time: O(N), Space: O(N)
        """
        # initialize DP array
        n = len(cost)
        dp = [0] * (n + 1)

        # start from the beginning
        # dp[0] & dp[1] == 0 -> no cost to start from 0 or 1
        for i in range(2, n + 1):
            # i-th item is the minimum cost to reach i
            # it's the minimum cost of i-1 plus cost[i-1] or i-2 plus cost[i-2]
            dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
          
        # return the last item 
        return dp[n]
