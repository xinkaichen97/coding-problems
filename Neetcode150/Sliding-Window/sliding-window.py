"""
Problems for Sliding Window
"""
from typing import List


class Solution:

    def maxProfit(self, prices: List[int]) -> int:
        """
        https://neetcode.io/problems/buy-and-sell-crypto
        Time: O(n), Space: O(1)
        """
        l, r = 0, 1
        profit = 0
        while r < len(prices):
            # if right is higher, update max profit
            if prices[l] < prices[r]:
                profit = max(profit, prices[r] - prices[l])
            # if not, we found the new lowest point
            else:
                l = r
            # increment r regardless
            r += 1
        return profit


    def lengthOfLongestSubstring(self, s: str) -> int:
        """
        https://neetcode.io/problems/longest-substring-without-duplicates
        Time: O(n), Space: O(1)
        """
        res = 0
        l = 0
        # window to keep unique items
        window = set()
        for r in range(len(s)):
            # while current is in window, remove left
            while s[r] in window:
                window.remove(s[l])
                l += 1
            # add to window and update max length
            window.add(s[r])
            length = r - l + 1
            res = max(res, length)
        return res


    def characterReplacement(self, s: str, k: int) -> int:
        """
        https://neetcode.io/problems/longest-repeating-substring-with-replacement
        Time: O(n), Space: O(1)
        """
        # count of letters in the window
        count = {}
        l = 0
        res = 0
        for r in range(len(s)):
            # update count
            count[s[r]] = count.get([s[r]], 0) + 1
            # current length minus the max counts is the number of replacements needed
            # if it's greater than k, move l and decrease count
            while (r - l + 1) - max(count.values()) > k:
                count[s[l]] -= 1
                l += 1
            # update res
            res = max(res, r - l + 1)
        return res

          
    def checkInclusion(self, s1: str, s2: str) -> bool:
        """
        https://neetcode.io/problems/permutation-string
        Time: O(n), Space: O(1)
        """
        # check equal lengths
        if len(s1) > len(s2):
            return False
        # arrays as counts
        count1, count2 = [0] * 26, [0] * 26
        n = len(s1)
        # update both counts until n
        for i in range(n):
            count1[ord(s1[i]) - ord('a')] += 1
            count2[ord(s2[i]) - ord('a')] += 1  
        # check a fixed window of length n
        for i in range(n, len(s2)):
            if count1 == count2:
                return True
            # decrease count for the previous left, and increase count for the new right
            count2[ord(s2[i]) - ord('a')] += 1
            count2[ord(s2[i - n]) - ord('a')] -= 1
        return count1 == count2
          
