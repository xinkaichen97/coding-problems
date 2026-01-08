"""
Problems for Bit Manipulation
"""
from typing import List


class Solution:

    def hammingWeight(self, n: int) -> int:
        """
        https://neetcode.io/problems/counting-bits/
        Time: O(1), Space: O(1)
        """
        res = 0
        while n:
            # use n & 1 to get the 1 bit if present
            # and then shift n to the right
            res += (n & 1)
            n >>= 1
        ## We can also do n & (n - 1) to eliminate exactly one 1 bit
        # while n:
        #     n &= n - 1
        #     res += 1
        return res

  
    def countBits(self, n: int) -> List[int]:
        """
        https://neetcode.io/problems/counting-bits/
        Time: O(n), Space: O(1)
        """
        offset = 1
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            # whenever i is the power of 2, update offset
            if offset * 2 == i:
                offset = i

            # dp logic: the lower numbers are the same, only need to add 1 to the highest bit
            # e.g. 4 -> 0100, dp[4] = 1 + dp[0]
            dp[i] = 1 + dp[i - offset]
          
        return dp
      
