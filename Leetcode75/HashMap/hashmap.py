"""
Problems for Hash maps/sets
"""
from typing import List


class Solution:
  
    def findDifference(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
        """
        https://leetcode.com/problems/find-the-difference-of-two-arrays/
        Time: O(N), Space: O(N)
        """
        res1, res2 = set(), set()
        for num in nums1:
            if num not in nums2:
                res1.add(num)
        for num in nums2:
            if num not in nums1:
                res2.add(num)
        return [list(res1), list(res2)]


    def uniqueOccurrences(self, arr: List[int]) -> bool:
        """
        https://leetcode.com/problems/unique-number-of-occurrences/
        Time: O(N), Space: O(N)
        """
        counts = {}
        for num in arr:
            counts[num] = counts.get(num, 0) + 1
        # check duplicates using a set
        return len(counts) == len(set(counts.values()))


    def closeStrings(self, word1: str, word2: str) -> bool:
        """
        https://leetcode.com/problems/determine-if-two-strings-are-close/
        Time: O(N), Space: O(N)
        """
        # two words have to share the same characters
        if set(word1) != set(word2):
            return False
        # get counts of each character
        counts1, counts2 = {}, {}
        for ch in word1:
            counts1[ch] = counts1.get(ch, 0) + 1
        for ch in word2:
            counts2[ch] = counts2.get(ch, 0) + 1
        # get counts of counts
        freq1, freq2 = {}, {}
        for count in counts1.values():
            freq1[count] = freq1.get(count, 0) + 1
        for count in counts2.values():
            freq2[count] = freq2.get(count, 0) + 1
        # if count frequencies are the same, it's possible to swap characters
        return freq1 == freq2


    def equalPairs(self, grid: List[List[int]]) -> int:
        """
        https://leetcode.com/problems/equal-row-and-column-pairs/
        Time: O(N^2), Space: O(N)
        """
        counts = defaultdict(int)
        res = 0
        # get count of each row
        for row in grid:
            counts[tuple(row)] += 1
        row = grid[0]
        # get each col and check if exists in counts
        for i in range(len(row)):
            col = [row[i] for row in grid]
            # add all counts to res
            if tuple(col) in counts:
                res += counts[tuple(col)]
        return res
      
