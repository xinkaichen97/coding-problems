# https://leetcode.com/problems/reverse-integer/

class Solution:
    # reverse string and convert to int
    def reverse(self, x: int) -> int:
        if x < 0:
            result = -1 * int(str(x)[1:][::-1])
        else:
            result = int(str(x)[::-1])
        if result not in range(-2 ** 31, 2 ** 31):
            return 0
        else:
            return result
     
     # Time limit exceeded
     # result = 0
     # while x != 0:
     #     result = result * 10 + x % 10
     #     x = x // 10
     # if result not in range(-2 ** 31, 2 ** 31):
     #     return 0
     # else:
     #     return result
