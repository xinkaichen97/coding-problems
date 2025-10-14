"""
Problems for Arrays and Hashing
"""
from typing import List


class Solution:
  
    def hasDuplicate(self, nums: List[int]) -> bool:
      """
      https://neetcode.io/problems/duplicate-integer
      Time: O(n), Space: O(n)
      """
      seen = {}
      for num in nums:
        if num in seen:
          return True
        seen.add(num)
      return True

  
    def isAnagram(self, s: str, t: str) -> bool:
      """
      https://neetcode.io/problems/is-anagram
      Time: O(n+m), Space: O(1)
      """
      if len(s) != len(t):
        return False
      count_s, count_t = {}, {}

      for i in range(len(s)):
        count_s[s[i]] = count_s.get(s[i], 0) + 1
        count_t[t[i]] = count_t.get(t[i], 0) + 1
      return count_s == count_t


    def twoSum(self, nums: List[int], target: int) -> List[int]:
      """
      https://neetcode.io/problems/two-integer-sum
      Time: O(n), Space: O(n)
      """
      seen = {}
      for i, num in enumerate(nums):
        diff = target - num
        if diff in seen:
          return [seen[diff], i]
        seen[num] = i

        
          
