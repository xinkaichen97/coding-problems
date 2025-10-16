"""
Problems for Arrays and Hashing
"""
from typing import List
from collections import defaultdict


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
        # return False if unequal lengths
        if len(s) != len(t):
          return False
        # counts for both strings
        count_s, count_t = {}, {}
        # update counts
        for i in range(len(s)):
          count_s[s[i]] = count_s.get(s[i], 0) + 1
          count_t[t[i]] = count_t.get(t[i], 0) + 1
        # directly compare two dictionaries
        return count_s == count_t


    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """
        https://neetcode.io/problems/two-integer-sum
        Time: O(n), Space: O(n)
        """
        seen = {}
        for i, num in enumerate(nums):
          # check if the diff is already stored
          diff = target - num
          if diff in seen:
            return [seen[diff], i]
          seen[num] = i


    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        """
        https://neetcode.io/problems/anagram-groups
        Time: O(n * m), Space: O(n) 
        -- m: max([len(s) for s in strs]), n: len(strs)
        """
        res = defaultdict(list)
        for s in strs:
          # for each string, keep counts
          counts = [0] * 26
          for ch in s:
            counts[ord(ch) - ord('a')] += 1
          # counts as the keys
          if tuple(counts) in res:
              # add the anagram to the list
              res[tuple(counts)].append(s)
        return res.values()


    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        """
        https://neetcode.io/problems/top-k-elements-in-list
        Time: O(n), Space: O(n)
        """
        counts = {}
        freq = defaultdict(list)
        res = []
        # get count of each num
        for num in nums:
            counts[num] = counts.get(num, 0) + 1
        # get mapping of each frequency
        for num, cnt in counts.items():
            freq[cnt].append(num)
        # start from n to 1 until k elements are found
        for i in range(len(nums), 0, -1):
            for num in freq[i]:
                res.append(num)
                if len(res) == k:
                    return res


    def encode(self, strs: List[str]) -> str:
        """
        https://neetcode.io/problems/string-encode-and-decode
        Time: O(m), Space: O(m + n) 
        -- m: sum([len(s) for s in strs]), n: len(strs)
        """
        res = ''
        # [app1e, Bee] -> 5#app1e3#Bee
        for s in strs:
            res += str(len(s)) + '#' + s
        return res

    def decode(self, s: str) -> List[str]:
        res = []
        i = 0
        while i < len(s):
            j = i
            # j stops at #
            while s[j] != '#':
                j += 1
            # find length and slice 
            length = int(s[i:j])
            res.append(s[j+1:j+1+length])
            i = j + 1 + length
        return res


    def productExceptSelf(self, nums: List[int]) -> List[int]:
        """
        https://neetcode.io/problems/products-of-array-discluding-self
        Time: O(n), Space: O(n)
        """
        # keep a prefix and a suffix array
        prefix = [1] * len(nums)
        suffix = [1] * len(nums)
        res = []
        # update prefix from the second index to the end
        for i in range(1, len(nums)):
            prefix[i] = prefix[i - 1] * nums[i - 1]
        # update suffix from the second last index to first
        for i in range(len(nums) - 2, -1, -1):
            suffix[i] = suffix[i + 1] * nums[i + 1]
        # multiply prefix and suffix for each index
        for pre, suf in zip(prefix, suffix):
            res.append(pre * suf)
        return res


    def isValidSudoku(self, board: List[List[str]]) -> bool:
        """
        https://neetcode.io/problems/valid-sudoku
        Time: O(n^2), Space: O(n^2)
        """
        # set for each col, row, and square
        cols = defaultdict(set)
        rows = defaultdict(set)
        squares = defaultdict(set)
        # go through every block
        for r in range(9):
            for c in range(9):
                if board[r][c] == ".":
                    continue
                # if the value already exists, return false
                # the key of squares is a tuple (r // 3, c // 3)
                if (board[r][c] in cols[c] or board[r][c] in rows[r] 
                    or board[r][c] in squares[(r // 3, c // 3)]):
                    return False
                # add new values
                cols[c].add(board[r][c])
                rows[r].add(board[r][c])
                squares[(r // 3, c // 3)].add(board[r][c])
        return True


    def longestConsecutive(self, nums: List[int]) -> int:
        """
        https://neetcode.io/problems/longest-consecutive-sequence
        Time: O(n), Space: O(n)
        """
        res = 0
        # create a set for nums
        numset = set(nums)
        for num in nums:
            # only start from the lower bound
            if num - 1 not in numset:
                length = 1
                # increment length while the num is found
                while num + length in numset:
                    length += 1
                    res = max(res, length)
        return res
                
