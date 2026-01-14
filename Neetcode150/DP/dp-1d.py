"""
Problems for Dynamic Programming
"""
from typing import List


class Solution:
  
    def climbStairs(self, n: int) -> int:
        """
        https://neetcode.io/problems/climbing-stairs
        Time: O(n), Space: O(n)
        """
        # base case
        if n <= 2:
            return n
        # create and initialize the dp array
        # the i-th item is the total number of distinct ways to reach step i
        dp = [0] * (n + 1)
        dp[1] = 1
        dp[2] = 2

        # the i-th item is just the sum of i - 1 and i - 2
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]

        # return the last item
        return dp[n]

        # ## Space optimized version
        # # one -> ways to reach the current step
        # # two -> ways to reach the previous step
        # one, two = 1, 1
        # for i in range(n - 1):
        #     one, two = one + two, one
        # return one

    ## top-down approach (recursion) with cache, same time/space complexity
    def climbStairsTopDown(self, n: int) -> int:
        # create an array to save results, -1 means not updated
        cache = [-1] * n
      
        # recursion
        def dfs(i):
            # base case: check if i == n, which means we can reach the last stair
            if i >= n:
                return i == n
            # return the cached value if already computed
            if cache[i] != -1:
                return cache[i]
            # find the answer recursively
            cache[i] = dfs(i + 1) + dfs(i + 2)
            return cache[i]

        return dfs(0)

  
    def rob(self, nums: List[int]) -> int:
        """
        https://neetcode.io/problems/house-robber
        Time: O(n), Space: O(n)
        """
        # base cases for empty and single-element arrays
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]

        # create and initialize the dp array
        # the i-th item is the max value up to the i-th house
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])

        # the max value is either the max of robbing the previous house
        # or robbing the current house
        for i in range(2, n):
            dp[i] = max(dp[i - 1], nums[i] + dp[i - 2])

        # return the last item in the dp array
        return dp[-1]

        # ## Space optimized version
        # # rob1 → best up to house i - 2
        # # rob2 → best up to house i - 1
        # rob1, rob2 = 0, 0
        # for num in nums:
        #     rob1, rob2 = rob2, max(num + rob1, rob2)
        # return rob2


    def rob2(self, nums: List[int]) -> int:
        """
        https://neetcode.io/problems/house-robber-ii
        Time: O(n), Space: O(n)
        """
        # base case
        if len(nums) == 1:
            return nums[0]

        # just the max results of nums[:-1] and nums[1:]
        return max(self.rob(nums[:-1]), self.rob(nums[1:]))
        

    def longestPalindrome(self, s: str) -> str:
        """
        https://neetcode.io/problems/longest-palindromic-substring
        Time: O(n^2), Space: O(n^2)
        """
        resIdx, resLen = 0, 0
        n = len(s)

        # ## could use two pointers to reduce space complexity to O(1)
        # for i in range(n):
        #     # odd length
        #     l, r = i, i
        #     while l >= 0 and r < len(s) and s[l] == s[r]:
        #         if (r - l + 1) > resLen:
        #             resIdx = l
        #             resLen = r - l + 1
        #         l -= 1
        #         r += 1
        #     # even length
        #     l, r = i, i + 1
        #     while l >= 0 and r < len(s) and s[l] == s[r]:
        #         if (r - l + 1) > resLen:
        #             resIdx = l
        #             resLen = r - l + 1
        #         l -= 1
        #         r += 1
        # return s[resIdx : resIdx + resLen]
      
        # initialize the 2-d dp array with False
        # dp[i][j] = True means s[i:j+1] is a palindrome
        dp = [[False] * n for _ in range(n)]

        # start from the bottom
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                # check if s[i] is equal to s[j], AND
                # if the substring is shorter than 3, OR s[i+1:j] is a palindrome
                if s[i] == s[j] and (j - i <= 2 or dp[i + 1][j - 1]):
                    dp[i][j] = True
                    # update res
                    if resLen < (j - i + 1):
                        resLen = (j - i + 1)
                        resIdx = i
        
        return s[resIdx : resIdx + resLen]


    def countSubstrings(self, s: str) -> int:
        """
        https://neetcode.io/problems/palindromic-substrings
        Time: O(n^2), Space: O(n^2)
        """
        # similar to the question above
        # could also use two pointers to reduce space complexity to O(1)
        res = 0
        n = len(s)
        
        # initialize the 2-d dp array with False
        # dp[i][j] = True means s[i:j+1] is a palindrome
        dp = [[False] * n for _ in range(n)]

        # start from the bottom
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                # check if s[i] is equal to s[j], AND
                # if the substring is shorter than 3, OR s[i+1:j] is a palindrome
                if s[i] == s[j] and (j - i <= 2 or dp[i + 1][j - 1]):
                    dp[i][j] = True
                    res += 1
    
        return res


    def numDecodings(self, s: str) -> int:
        """
        https://neetcode.io/problems/decode-ways
        Time: O(n), Space: O(n)
        """
        # initialize the dp dict
        dp = {len(s): 1}

        # bottom-up
        for i in range(len(s) - 1, -1, -1):
            # base case: invalid decoding
            if s[i] == "0":
                dp[i] = 0
            else:
                dp[i] = dp[i + 1]
            # if i + 1 is in bounds, and the two letters are between 10 and 26
            # add two-letter decoding
            if i + 1 < len(s) and (s[i] == "1" or 
                s[i] == "2" and s[i + 1] in "0123456"):
                dp[i] += dp[i + 2]
        
        return dp[0]
      

    def coinChange(self, coins: List[int], amount: int) -> int:
        """
        https://neetcode.io/problems/coin-change
        Time: O(n * a), Space: O(a)
        """
        # initialize the dp array as a big number
        # dp[i] -> minimum coins needed to make amount i
        dp = [amount + 1] * (amount + 1)
        dp[0] = 0

        # loop through all the values between 1 and amount
        for val in range(1, amount + 1):
            # check every coin that is smaller than the current value
            for coin in coins:
                if val >= coin:
                    # need just 1 coin to make val from (val - coin)
                    dp[val] = min(dp[val], 1 + dp[val - coin])

        # return the last item if the result was updated
        return dp[amount] if dp[amount] != amount + 1 else -1
      
