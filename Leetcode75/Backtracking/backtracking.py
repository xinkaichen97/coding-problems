"""
Problems for Backtracking
"""
from typing import List


class Solution:
  
    def letterCombinations(self, digits: str) -> List[str]:
        """
        https://leetcode.com/problems/letter-combinations-of-a-phone-number
        Time: O(n * 4^n), Space: O(n), n - len(digits)
        """
        if not digits:
            return []
          
        # number to letter mapping
        mapping = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz',
        }
        res = []

        # backtrack function
        def backtrack(idx, comb):
            # return criterion: indices reach the end
            if idx == len(digits):
                res.append(comb)
                return

            # call backtrack with index + 1 and updated letters
            # for each letter in digits, the worst case is 4 possible next letters, so the total path is 4^n
            # and since string concatenation is O(n), total time complexity is O(n * 4^n)
            for letter in mapping[digits[idx]]:
                backtrack(idx + 1, comb + letter)

        # backtrack from index 0 and an empty string
        backtrack(0, "")

        return res

  
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        """
        https://leetcode.com/problems/combination-sum-iii
        Time: O(k * C(9,k)), Space: O(k)
        """
        res = []
        ans = []

        # backtracking function
        def backtrack(currsum, idx, k, n):
            # return criterion 1: if the current sum is already greater than n (pruning)
            if currsum > n:
                return
            # return criterion 2: if the current list has length k and sum is n
            # each operation is O(k)
            if len(ans) == k:
                if currsum == n:
                    res.append(ans.copy())
                return
              
            # index from 1 to 9, but we only need (k - len(ans)) more items
            # we select k numbers from 1 to 9, so the upper bound of total paths is C(9,k)
            for i in range(idx, 9 - (k - len(ans)) + 2):
                # update current sum and add to ans
                currsum += i
                ans.append(i)

                # call recursive function with index + 1
                backtrack(currsum, i + 1, k, n)

                # backtrack: subtract num from currsum and remove from ans
                currsum -= i
                ans.pop()
              
        # backtrack from 1      
        backtrack(0, 1, k, n)
      
        return res
              
