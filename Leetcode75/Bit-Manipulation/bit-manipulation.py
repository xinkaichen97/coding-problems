"""
Problems for Bit Manipulation
"""
from typing import List


class Solution(object):
  
    def countBits(self, n: int) -> List[int]:
        """
        https://leetcode.com/problems/counting-bits
        Time: O(n), Space: O(1)
        """
        # create a DP array
        res = [0] * (n + 1)
        for x in range(1, n + 1):
            # count(x) = count(x / 2) + x % 2
            # x // 2 is x >> 1 and x % 2 is x & 1
            res[x] = res[x >> 1] + (x & 1) 

        return res 
