"""
Solutions for Binary Search problems
"""
from typing import List


class Solution:
  
    def search(self, nums: List[int], target: int) -> int:
      """
      https://neetcode.io/problems/binary-search
      Iterative solution
      Time: O(logn), Space: O(1)
      """
      l, r = 0, len(nums) - 1
      while l <= r:
        # (l + r) // 2 can lead to overflow  
        m = l + (r - l) // 2
        if nums[m] == target:
            return m
        elif nums[m] < target:
            l = m + 1
        else:
            r = m - 1
      return -1


    def search_recursive(self, nums: List[int], target: int) -> int:
      """
      https://neetcode.io/problems/binary-search
      Recursive solution
      Time: O(logn), Space: O(logn)
      """
      return self.binary_search(nums, 0, len(nums) - 1, target)

  
    def binary_search(self, nums: List[int], l: int, r: int, target: int) -> int:
      # base case to avoid max recursion
      if l > r:
          return -1
      m = l + (r - l) // 2
      if nums[m] == target:
          return m
      elif nums[m] < target:
          return self.binary_search(nums, m + 1, r, target)
      else:
          return self.binary_search(nums, l, m - 1, target)


    def findMin(self, nums: List[int]) -> int:
      """
      https://neetcode.io/problems/find-minimum-in-rotated-sorted-array
      Time: O(logn), Space: O(1)
      """
      l, r = 0, len(nums) - 1
      res = nums[0]
      while l <= r:
        # if the left is smaller than the right, it's already sorted
        # compare the leftmost number to res
        if nums[l] < nums[r]:
          res = min(res, nums[l])
          break
        # find the mid point and compare with res, it could be the minimum
        m = l + (r - l) // 2
        res = min(res, nums[m])
        # if left is smaller than mid, the turning point is in the other half
        if nums[l] <= nums[m]:
          l = m + 1
        # if left is greater than mid, the turning point must be in this half
        else:
          r = m - 1
      return res


    def search(self, nums: List[int], target: int) -> int:
      """
      https://neetcode.io/problems/find-target-in-rotated-sorted-array
      Time: O(logn), Space: O(1)
      """
      l, r = 0, len(nums) - 1
      while l <= r:
        m = l + (r - l) // 2
        if nums[m] == target:
            return m
        # if left is smaller than mid, the left half is sorted
        if nums[l] <= nums[m]:
          # if target not in the sorted left half, search right
          if nums[l] > target or nums[m] < target
            l = m + 1
          # otherwise search within this half
          else:
            r = m - 1
        # otherwise the right half is sorted
        else:
          # if target not in the sorted right half, search left
          if nums[r] < target or nums[m] > target:
            r = m - 1
          # otherwise search within this half
          else:
            l = m + 1
      return -1


    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
      """
      https://neetcode.io/problems/search-2d-matrix
      Time: O(logm+logn) = O(log(m*n)), Space: O(1)
      """
      # find row from top and bottom
      t, b = 0, len(matrix) - 1
      while t <= b:
          row = (t + b) // 2
          if matrix[row][-1] < target:
              t = row + 1
          elif matrix[row][0] > target:
              b = row - 1
          else:
              break
      # if none of the row matches, return False
      if t > b:
          return False
      # find target within the target row
      l, r = 0, len(matrix[0]) - 1
      while l <= r:
          m = (l + r) // 2
          if matrix[row][m] == target:
              return True
          elif matrix[row][m] < target:
              l = m + 1
          else:
              r = m - 1
      return False
            

class TimeMap:
    """
    https://neetcode.io/problems/time-based-key-value-store
    Time: O(1) for set(), O(logn) for get(), Space: O(m * n), m - len(self.mapping), n - len(self.mapping[key])
    """

    def __init__(self):
        self.mapping = defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        # add [timestamp, value] pairs
        self.mapping[key].append([timestamp, value])

    def get(self, key: str, timestamp: int) -> str:
        res = ""
        values = self.mapping.get(key, [])
        # use binary search for log(n) time
        l = 0
        r = len(values) - 1
        # either use [l, r] (while l <= r) or [l, r) (while l < r)
        # r will be m - 1 or m
        while l <= r:
            m = (l + r) // 2
            # keep updating the rightmost index and res (bisect_left)
            # if use < instead of <=, it becomes bisect_left
            if values[m][0] <= timestamp:
                res = values[m][1]
                l = m + 1
            else:
                r = m - 1
        return res
      
