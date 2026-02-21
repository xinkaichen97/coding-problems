"""
Problems for Greedy
"""
from typing import List


class Solution:

    def maxSubArray(self, nums: List[int]) -> int:
        """
        https://neetcode.io/problems/maximum-subarray
        Time: O(n), Space: O(n) for dp, O(1) for Kadane
        """
        # # DP
        # # dp[i] = maximum sum of subarray ending at index i
        # dp = [0] * len(nums)
        # dp[0] = nums[0]
        
        # for i in range(1, n):
        #     # two choices: extend previous subarray or start new subarray
        #     dp[i] = max(nums[i], dp[i-1] + nums[i])
        # return max(dp)
      
        # Kadane's algorithm
        res = nums[0]
        curSum = 0
        for num in nums:
            # if curSum is negative, it means we need to start from the current position
            if curSum < 0:
                curSum = 0
            curSum += num
            res = max(res, curSum)
        return res
    
    
    def canJump(self, nums: List[int]) -> bool:
        """
        https://neetcode.io/problems/jump-game
        Time: O(n), Space: O(n)
        """
        # # DP - O(n^2)
        # # dp[i] = whether it can reach the end from index i
        # n = len(nums)
        # dp = [False] * n
        # dp[-1] = True
      
        # for i in range(n - 2, -1, -1):
        #     end = min(n, i + nums[i] + 1) # prevent overflow
        #     # check if any of 
        #     for j in range(i + 1, end):
        #         if dp[j]:
        #             dp[i] = True
        #             break
        # return dp[0]
        
        # greedy - O(n)
        goal = len(nums) - 1
        for i in range(len(nums) - 2, -1, -1):
            # update goal whenever it can be reached from i
            if i + nums[i] >= goal:
                goal = i
        return goal == 0
      

    def jump(self, nums: List[int]) -> int:
        """
        https://neetcode.io/problems/jump-game-ii
        Time: O(n), Space: O(1)
        """
        # # DP - O(n^2)
        # dp = [float("inf")] * len(nums)
        # dp[-1] = 0
        # for i in range(len(nums) - 2, -1, -1):
        #     end = min(len(nums), i + nums[i] + 1)
        #     for j in range(i + 1, end):
        #         dp[i] = min(dp[i], 1 + dp[j])
        # return dp[0]
        
        res = 0
        l, r = 0, 0
        while r < len(nums) - 1:
            # the farthest it can jump from the current range [l, r]
            nxt = r
            while l <= r:
                nxt = max(nxt, l + nums[l])
                l += 1
            # go to the next range [r + 1, nxt]
            l = r + 1
            r = nxt
            # increment res as we take greedy steps every time
            res += 1
            
        return res
        
