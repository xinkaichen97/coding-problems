"""
Problems for Bit Manipulation
"""
from typing import List


class Solution:

    def hammingWeight(self, n: int) -> int:
        """
        https://neetcode.io/problems/number-of-one-bits
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
        https://neetcode.io/problems/counting-bits
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


    def reverseBits(self, n: int) -> int:
        """
        https://neetcode.io/problems/reverse-bits
        Time: O(1), Space: O(1)
        """
        res = 0
        for i in range(32):
            # get the i-th bit of n
            bit = (n >> i) & 1
            # put it at position (31 - i), use OR to avoid affecting other bits
            res = res | (bit << (31 - i))
            
        return res


    def missingNumber(self, nums: List[int]) -> int:
        """
        https://neetcode.io/problems/missing-number
        Time: O(n), Space: O(1)
        """
        res = len(nums)
        for i in range(len(nums)):
            # for each index and number, do XOR until there's one number left
            res ^= (i ^ nums[i])
        return res


    def getSum(self, a: int, b: int) -> int:
        """
        https://neetcode.io/problems/sum-of-two-integers
        Time: O(1), Space: O(1)
        """
        res = 0
        carry = 0
        mask = 0xFFFFFFFF # all 1's
        max_int = 0x7FFFFFFF # max positive int, 011...11

        for i in range(32):
            # get i-th bits from a and b
            a_bit = (a >> i) & 1
            b_bit = (b >> i) & 1
            # use XOR to get the current bit after carry
            bit = a_bit ^ b_bit ^ carry
            # calculate if there's a carry
            carry = (a_bit + b_bit + carry) >= 2
            # add the current bit to res
            if bit:
                res |= (1 << i)

        # handle negative numbers in Python
        if res > max_int:
            # use mask to flip all bits and get (magnitude - 1)
            # then do a NOT operation to get the correct negative number
            res = ~(res ^ mask)
            ## equivalent: minus 2^32
            # res = res - 0x100000000
        
        return res
        

    def reverse(self, x: int) -> int:
        """
        https://neetcode.io/problems/reverse-integer
        Time: O(1), Space: O(1)
        """
        # keep track of min and max
        MIN = -2147483648  # -2^31
        MAX = 2147483647  #  2^31-1
        res = 0
        while x:
            # use math.fmod instead of % to handle negative values, e.g. -1 % 10 = 9, WRONG!
            digit = int(math.fmod(x, 10))
            # use float division and convert to int to handle negative values, e.g. -1 // 10 = -1
            x = int(x / 10)
            # before updating res, if it's already greater than MAX or smaller than MIN, overflow -> return 0
            # sme if the current digit is greater than MAX's final digit or smaller than MIN's final digit
            if res > MAX // 10 or (res == MAX // 10 and digit > MAX % 10):
                return 0
            if res < MIN // 10 or (res == MIN // 10 and digit < MIN % 10):
                return 0
            # update res
            res = res * 10 + digit
            
        return res
        
