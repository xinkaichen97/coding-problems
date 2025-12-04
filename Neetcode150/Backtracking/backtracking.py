"""
Problems for Backtracking
"""
from typing import List


class Solution:
  
    def combinationSum(self, nums: List[int], target: int) -> List[List[int]]:
        """
        https://neetcode.io/problems/combination-target-sum
        Time: O(2^(t/m)), Space: O(t/m), t - target, m - min(nums)
        """
        res = []

        # backtracking decision tree: include or not include certain number(s)
        # the maximum height is target / minimum value in nums, and each node has two branches - 2 ^ t/m
        def backtrack(i, curr, total):
            # if target is reached, add a copy of the current list
            if total == target:
                res.append(curr.copy())
                return
              
            # if total is already greater or index out of bounds, stop
            if i >= len(nums) or total > target:
                return

            # include: add current number and backtrack with updated sum
            curr.append(nums[i])
            backtrack(i, curr, total + nums[i])
            # not include: go to the next number
            curr.pop()
            backtrack(i + 1, curr, total)
        
        backtrack(0, [], 0)
        return res


    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        https://neetcode.io/problems/combination-target-sum-ii
        Time: O(n * 2^n), Space: O(n), n - len(candidates)
        """
        res = []
        # sort the array to make sure that the same number
        candidates.sort()

        # backtracking decision tree: include or not include certain value (and its duplicates)
        # the maximum height is n - total 2^n paths
        def backtrack(i, curr, total):
            # if target is reached, add a copy of the current list
            if total == target:
                res.append(curr.copy())
                return

            # if total is already greater or index out of bounds, stop
            if i == len(candidates) or total > target:
                return

            # include the current value
            # string operation is O(n)
            curr.append(candidates[i])
            backtrack(i + 1, curr, total + candidates[i])
            curr.pop()
            # do not include the current value, while loop to go to the next one
            while i + 1 < len(candidates) and candidates[i] == candidates[i + 1]:
                i += 1
            backtrack(i + 1, curr, total)
        
        backtrack(0, [], 0)
        return res
      
