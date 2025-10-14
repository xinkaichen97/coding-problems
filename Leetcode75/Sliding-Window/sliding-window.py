"""
Problems for Sliding Windows
"""


class Solution:
  
    def findMaxAverage(self, nums: List[int], k: int) -> float:
      """
      643. Maximum Average Subarray I
      """
      # get initial sum
      currsum = res = sum(nums[:k])
      for i in range(k, len(nums)):
          # add the new one, subtract the old one
          currsum += nums[i] - nums[i - k]
          if currsum >= ans:
              ans = currsum
      return res / k


    
