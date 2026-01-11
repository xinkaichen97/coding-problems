"""
Problems for Dynamic Programming
"""


class Solution:
  
    def climbStairs(self, n: int) -> int:
        """
        https://neetcode.io/problems/climbing-stairs
        Time: O(n), Space: O(n)
        """
        # base case
        if n <= 2:
            return n
        # create and initialize the dp array
        # the i-th item is the total number of distinct ways to reach step i
        dp = [0] * (n + 1)
        dp[1] = 1
        dp[2] = 2

        # the i-th item is just the sum of i - 1 and i - 2
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]

        # return the last item
        return dp[n]

        # ## Space optimized version
        # # one -> ways to reach the current step
        # # two -> ways to reach the previous step
        # one, two = 1, 1
        # for i in range(n - 1):
        #     one, two = one + two, one
        # return one


    def rob(self, nums: List[int]) -> int:
        """
        https://neetcode.io/problems/house-robber
        Time: O(n), Space: O(n)
        """
        # base cases for empty and single-element arrays
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]

        # create and initialize the dp array
        # the i-th item is the max value up to the i-th house
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])

        # the max value is either the max of robbing the previous house
        # or robbing the current house
        for i in range(2, n):
            dp[i] = max(dp[i - 1], nums[i] + dp[i - 2])

        # return the last item in the dp array
        return dp[-1]

        # ## Space optimized version
        # # rob1 → best up to house i - 2
        # # rob2 → best up to house i - 1
        # rob1, rob2 = 0, 0
        # for num in nums:
        #     rob1, rob2 = rob2, max(num + rob1, rob2)
        # return rob2


    def rob2(self, nums: List[int]) -> int:
        """
        https://neetcode.io/problems/house-robber-ii
        Time: O(n), Space: O(n)
        """
        # base case
        if len(nums) == 1:
            return nums[0]

        # just the max results of nums[:-1] and nums[1:]
        return max(self.rob(nums[:-1]), self.rob(nums[1:]))
        
