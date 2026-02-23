"""
Problems for Math
"""
from typing import List


class Solution:

    def isHappy(self, n: int) -> bool:
        """
        https://neetcode.io/problems/non-cyclical-number
        Time: O(logn), Space: O(logn)
        """
        # use a HashSet to track if a number has become cyclical
        seen = set()
        # a happy number would become 1 eventually
        while n != 1:
            n = self.sumOfSquares(n)
            # if n is already seen, it's not a happy number; otherwise add it to seen
            if n in seen:
                return False
            seen.add(n)
        return True

        # # fast and slow pointers approach - O(1) space
        # slow, fast = n, self.sumOfSquares(n)
        # while slow != fast:
        #     fast = self.sumOfSquares(fast)
        #     fast = self.sumOfSquares(fast)
        #     slow = self.sumOfSquares(slow)
        # return True if fast == 1 else False

  
    def sumOfSquares(self, n: int) -> int:
        """
        Helper function to get the sum of squares of each digit
        """
        res = 0
        while n:
            res += (n % 10) ** 2
            n = n // 10
        return res
      

    def plusOne(self, digits: List[int]) -> List[int]:
        """
        https://neetcode.io/problems/plus-one
        Time: O(n), Space: O(n)
        """
        for i in range(len(digits) - 1, -1, -1):
            # if not carry, add one and return
            if digits[i] < 9:
                digits[i] += 1
                return digits
            # if the digit is 9, we keep going with a carry of one
            digits[i] = 0
            
        return [1] + digits
      

    def myPow(self, x: float, n: int) -> float:
        """
        https://neetcode.io/problems/pow-x-n
        Time: O(logn), Space: O(logn) for recursive, O(1) for iterative
        """
        # base cases
        if x == 0:
            return 0
        if n == 0:
            return 1

        # recursive solution
        # if negative, turn x into 1/x
        if n < 0:
            x, n = 1 / x, -n

        # to speed up, x^n = (x^2)^(n/2)
        # if n is odd, need to multiply another x
        res = self.myPow(x * x, n // 2)
        return x * res if n % 2 else res

        # # iterative solution
        # res = 1
        # power = abs(n)
        # while power:
        #     # when the power is odd, times x
        #     if power % 2:
        #         res *= x
        #     x *= x
        #     power //= 2
        # return res if n > 0 else 1 / res

  
    def multiply(self, num1: str, num2: str) -> str:
        """
        https://neetcode.io/problems/multiply-strings
        Time: O(m * n), Space: O(m + n)
        """
        # base case: return 0 if exists
        if "0" in [num1, num2]:
            return "0"

        # the max length is m + n
        res = [0] * (len(num1) + len(num2))
        # reverse strings for easier looping
        num1, num2 = num1[::-1], num2[::-1]

        for i in range(len(num1)):
            for j in range(len(num2)):
                # calculate the current product
                prod = int(num1[i]) * int(num2[j])
                # directly add to res[i+j], calculate carry for res[i+j+1], and then keep the last digit
                res[i + j] += prod
                res[i + j + 1] += res[i + j] // 10
                res[i + j] %= 10

        # reverse it back and remove leading zeros
        res = res[::-1]
        for i in range(len(res)):
            if res[i] != 0:
                res = res[i:]
                break

        # convert to string
        return "".join(map(str, res))
