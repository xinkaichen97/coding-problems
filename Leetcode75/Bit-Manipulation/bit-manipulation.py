"""
Problems for Bit Manipulation
"""
from typing import List


class Solution:
  
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


    def singleNumber(self, nums: List[int]) -> int:
        """
        https://leetcode.com/problems/single-number
        Time: O(n), Space: O(1)
        """
        res = 0
        
        # perform XOR on all numbers, the rest is the single one
        for i in nums:
            res ^= i
          
        return res


    def minFlips(self, a: int, b: int, c: int) -> int:
        """
        https://leetcode.com/problems/minimum-flips-to-make-a-or-b-equal-to-c
        Time: O(n), Space: O(1)
        """
        res = 0
      
        # calculate until all three become zero
        while a or b or c:
            # if c's last bit is 1, only flip if a or b last bits are both 0's
            if c & 1:
                res += 0 if ((a & 1) or (b & 1)) else 1
            # if c's last bit is 0, a and b last bits must both be zero
            else:
                res += (a & 1) + (b & 1)
            
            # shift all numbers to the right to check the next bits
            a >>= 1
            b >>= 1
            c >>= 1
            
        return res
      
