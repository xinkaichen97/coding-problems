from typing import List
from collections import defaultdict


class Solution:
  
    def moveZeroes(self, nums: List[int]) -> None:
      """
      283. Move Zeroes
      Do not return anything, modify nums in-place instead.
      Time: O(n), Space: O(1)
      """
      curr = 0
      for i in range(len(nums)):
        # swap non-zero item with current zero position
        if nums[i] != 0:
          nums[curr], nums[i] = nums[curr], nums[i]
          # increment curr so that it moves towards the next zero
          curr += 1

  
    def isSubsequence(self, s: str, t: str) -> bool:
      """
      392. Is Subsequence
      Time: O(n), Space: O(1)
      """
      i = j = 0
      while i < len(t) and j < len(s):
        # increment j if matched
        if t[i] == s[j]:
          j += 1
        i += 1
      # check if j reaches the end of s
      return j == len(s)

  
    def maxArea(self, height: List[int]) -> int:
      """
      11. Container With Most Water
      Time: O(n), Space: O(1)
      """
      l, r = 0, len(height) - 1
      res = 0
      while l < r:
        # calculate area with current l and r
        area = (r - l) * min(height[l], height[r])
        res = max(res, area)
        # increment l if height[l] is lower, otherwise decrement r
        if height[l] <= height[r]:
          l += 1
        else:
          r -= 1
      return res

  
    def maxOperations(self, nums: List[int], k: int) -> int:
      """
      1679. Max Number of K-Sum Pairs
      Time: O(n), Space: O(n)
      """
      res = 0
      # hashmap to keep counts
      seen = defaultdict(int)
      for num in nums:
        diff = k - num
        # decrement count of diff if a pair is found
        if seen[diff] > 0:
          res += 1
          seen[diff] -= 1
        else:
          # increment count if a pair is not yet found
          seen[num] += 1
      return res

    def maxOperations2(self, nums: List[int], k: int) -> int:
      """
      1679. Max Number of K-Sum Pairs
      Time: O(nlogn), Space: O(1)
      """
      # sort the array and start from both ends
      nums.sort()
      res = 0
      l, r = 0, len(nums) - 1
      while l < r:
        if nums[l] + nums[r] == k:
          res += 1
          l += 1
          r -= 1
        elif nums[l] + nums[r] < k:
          l += 1
        else:
          r -= 1
      return res
