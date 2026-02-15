"""
Problems for Backtracking
"""
from typing import List


class Solution:

    def subsets(self, nums: List[int]) -> List[List[int]]:
        """
        https://neetcode.io/problems/subsets
        Time: O(n * 2^n), Space: O(n)
        """
        res = []
        def backtrack(subset, i):
            # return condition: if i reaches the end
            if i >= len(nums):
                res.append(subset.copy())
                return
              
            # include the current num
            subset.append(nums[i])
            backtrack(subset, i + 1)

            # not include the current num
            subset.pop()
            backtrack(subset, i + 1)

        backtrack([], 0)
        return res

  
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


    def permute(self, nums: List[int]) -> List[List[int]]:
        """
        https://neetcode.io/problems/permutations
        Time: O(n! * n), Space: O(n! * n)
        """
        res = []

        def backtrack(nums, idx):
            # return condition: idx reaches the end
            if idx == len(nums):
                res.append(nums.copy())
                return
                
            # keep 0 to idx-1 the same, and update from idx to end
            for i in range(idx, len(nums)):
                # swap numbers at each index with idx
                nums[i], nums[idx] = nums[idx], nums[i]
                # recursively build the answer with idx + 1
                backtrack(nums, idx + 1)
                # backtrack: swap the numbers back
                nums[i], nums[idx] = nums[idx], nums[i]
        
        backtrack(nums, 0)

        # # approach 2: use a pick array to mark which elements are already used
        # def backtrack(perms, nums, pick):
        #     if len(perms) == len(nums):
        #         res.append(perms.copy())
        #         return
        #     for i in range(len(nums)):
        #         if not pick[i]:
        #             perms.append(nums[i])
        #             pick[i] = True
        #             backtrack(perms, nums, pick)
        #             perms.pop()
        #             pick[i] = False
        # backtrack([], nums, [False] * len(nums))
        
        return res


    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        """
        https://neetcode.io/problems/subsets-ii
        Time: O(n * 2^n), Space: O(2^n)
        """
        nums.sort()
        res = []
        def backtrack(subset, i):
            if i >= len(nums):
                res.append(subset.copy())
                return
            
            subset.append(nums[i])
            backtrack(subset, i + 1)
            subset.pop()
            
            # only difference from Subset 
            # if not including the current num, skip all duplicates
            while i + 1 < len(nums) and nums[i] == nums[i + 1]:
                i += 1
            backtrack(subset, i + 1)
        
        backtrack([], 0)
        return res

    
    def exist(self, board: List[List[str]], word: str) -> bool:
        """
        https://neetcode.io/problems/search-for-word
        Time: O(m * 4^n), Space: O(n), m - total cells, n - len(word)
        """
        n_row, n_col = len(board), len(board[0])
        seen = set()

        
        def backtrack(r, c, i):
            # base case: return true if it reaches the end of the word
            if i == len(word):
                return True

            # if the board constraint is violated or the cell is already checked, return false
            if min(r, c) < 0 or r >= n_row or c >= n_col or word[i] != board[r][c] or (r, c) in seen:
                return False

            # add current cell to the set
            seen.add((r, c))
            # check all four directions
            res = backtrack(r + 1, c, i + 1) or backtrack(r, c + 1, i + 1) or backtrack(r - 1, c, i + 1) or backtrack(r, c - 1, i + 1)
            # and then remove the current cell
            seen.remove((r, c))
            return res

        # backtrack from every cell in the board, return true immediately if found
        for r in range(n_row):
            for c in range(n_col):
                if backtrack(r, c, 0):
                    return True
        
        return False
