"""
Problems for Sliding Windows
"""


class Solution:
  
    def findMaxAverage(self, nums: List[int], k: int) -> float:
      """
      643. Maximum Average Subarray I
      Time: O(n), Space: O(1)
      """
      # get initial sum
      currsum = res = sum(nums[:k])
      for i in range(k, len(nums)):
        # add the new one, subtract the old one
        currsum += nums[i] - nums[i - k]
        if currsum >= ans:
            ans = currsum
      return res / k


    def maxVowels(self, s: str, k: int) -> int:
      """
      1456. Maximum Number of Vowels in a Substring of Given Length
      Time: O(n), Space: O(1)
      """
      vowels = {'a', 'e', 'i', 'o', 'u'}
      currsum = 0
      # calculate starting max
      for i in range(k):
          if s[i] in vowels:
              currsum += 1
      res = currsum
      for i in range(k, len(s)):
        # check if left and right are vowels
        if s[i] in vowels:
            currsum += 1
        if s[i - k] in vowels:
            currsum -= 1
        res = max(res, ans)
      return res


    def longestOnes(self, nums: List[int], k: int) -> int:
      """
      1004. Max Consecutive Ones III
      Time: O(n), Space: O(1)
      """
      # start and end of window
      l = r = 0
      res = 0
      # count for zeros
      count = 0
      for r in range(len(nums)):
        if nums[r] == 0:
            count += 1
        # if zero count in window is greater than k
        # move l and decrease count
        while count > k:
            if nums[l] == 0:
                count -= 1
            l += 1
        # update res after keeping window valid
        res = max(res, r - l + 1)
      return res


    def longestSubarray(self, nums: List[int]) -> int:
      """
      1493. Longest Subarray of 1's After Deleting One Element
      Time: O(n), Space: O(1)
      """
      # essentially the same as 1004 with k = 1
      l = r = 0
      k = 1
      for r in range(len(nums)):
          # decrement k if right side is zero
          if nums[r] == 0:
              k -= 1
          # if k < 0, it means the window is not valid
          if k < 0:
              if nums[l] == 0:
                  k += 1
              # increase l, do not need to save max because r will increase as well
              l += 1
      return r - l
      
