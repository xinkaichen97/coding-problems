"""
Problems for Greedy
"""
from typing import List


class Solution:

    def maxSubArray(self, nums: List[int]) -> int:
        """
        https://neetcode.io/problems/maximum-subarray
        Time: O(n), Space: O(n) for dp, O(1) for Kadane
        """
        # # DP
        # # dp[i] = maximum sum of subarray ending at index i
        # dp = [0] * len(nums)
        # dp[0] = nums[0]
        
        # for i in range(1, n):
        #     # two choices: extend previous subarray or start new subarray
        #     dp[i] = max(nums[i], dp[i-1] + nums[i])
        # return max(dp)
      
        # Kadane's algorithm
        res = nums[0]
        curSum = 0
        for num in nums:
            # if curSum is negative, it means we need to start from the current position
            if curSum < 0:
                curSum = 0
            curSum += num
            res = max(res, curSum)
        return res
    
    
    def canJump(self, nums: List[int]) -> bool:
        """
        https://neetcode.io/problems/jump-game
        Time: O(n), Space: O(n)
        """
        # # DP - O(n^2)
        # # dp[i] = whether it can reach the end from index i
        # n = len(nums)
        # dp = [False] * n
        # dp[-1] = True
      
        # for i in range(n - 2, -1, -1):
        #     end = min(n, i + nums[i] + 1) # prevent overflow
        #     # check if any of 
        #     for j in range(i + 1, end):
        #         if dp[j]:
        #             dp[i] = True
        #             break
        # return dp[0]
        
        # greedy - O(n)
        goal = len(nums) - 1
        for i in range(len(nums) - 2, -1, -1):
            # update goal whenever it can be reached from i
            if i + nums[i] >= goal:
                goal = i
        return goal == 0
      

    def jump(self, nums: List[int]) -> int:
        """
        https://neetcode.io/problems/jump-game-ii
        Time: O(n), Space: O(1)
        """
        # # DP - O(n^2)
        # dp = [float("inf")] * len(nums)
        # dp[-1] = 0
        # for i in range(len(nums) - 2, -1, -1):
        #     end = min(len(nums), i + nums[i] + 1)
        #     for j in range(i + 1, end):
        #         dp[i] = min(dp[i], 1 + dp[j])
        # return dp[0]
        
        res = 0
        l, r = 0, 0
        while r < len(nums) - 1:
            # the farthest it can jump from the current range [l, r]
            nxt = r
            while l <= r:
                nxt = max(nxt, l + nums[l])
                l += 1
            # go to the next range [r + 1, nxt]
            l = r + 1
            r = nxt
            # increment res as we take greedy steps every time
            res += 1
            
        return res
        

    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        """
        https://neetcode.io/problems/gas-station
        Time: O(n), Space: O(1)
        """
        # # two-pointer approach
        # start, end = len(gas) - 1, 0
        # tank = gas[start] - cost[start]
        # while start > end:
        #     if tank < 0:
        #         start -= 1
        #         tank += gas[start] - cost[start]
        #     else:
        #         tank += gas[end] - cost[end]
        #         end += 1
        # return start if tank >= 0 else -1
        
        # immediately return if total gas is lower than total cost
        if sum(gas) < sum(cost):
            return -1
            
        total = 0
        res = 0
        for i in range(len(gas)):
            total += (gas[i] - cost[i])
            # if accumulated total is negative
            # that means there's no valid starting point from res to i, go to i + 1
            if total < 0:
                total = 0
                res = i + 1
                
        return res


    def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
        """
        https://neetcode.io/problems/hand-of-straights
        Time: O(nlogn), Space: O(n)
        """
        # if hand not divisible by groupSize, directly return false
        if len(hand) % groupSize:
            return False

        # sort the array and go through each number
        hand.sort()
        counts = Counter(hand) # Counter handles missing keys
        for num in hand:
            # if current count > 0, try to start a group
            if counts[num]:
                for i in range(num, num + groupSize):
                    if not counts[i]:
                        return False
                    # decrement when included in a group
                    counts[i] -= 1

        # # O(n) solution - sorting not needed
        # for num in hand:
        #     start = num
        #     # for the current number, try to go back as much as possible
        #     while counts[start - 1]:
        #         start -= 1
        #     # try to start a group
        #     while start <= num:
        #         # for multiple groups at the same start, keep searching until count is 0
        #         while counts[start] > 0:
        #             for i in range(start, start + groupSize):
        #                 if not counts[i]:
        #                     return False
        #                 counts[i] -= 1
        #         start += 1
                
        return True


    def mergeTriplets(self, triplets: List[List[int]], target: List[int]) -> bool:
        """
        https://neetcode.io/problems/merge-triplets-to-form-target
        Time: O(n), Space: O(1)
        """
        # three boolean values to track each position
        x = y = z = False
        for t in triplets:
            # skip if any value is greater than target
            if t[0] > target[0] or t[1] > target[1] or t[2] > target[2]:
                continue
            # update x, y, z using OR
            x |= t[0] == target[0]
            y |= t[1] == target[1]
            z |= t[2] == target[2]
            if x and y and z:
                return True
                
        return False


    def partitionLabels(self, s: str) -> List[int]:
        """
        https://neetcode.io/problems/partition-labels
        Time: O(n), Space: O(1)
        """
        # use a HashMap to save the last index of a character
        mp = {}
        for i, c in enumerate(s):
            mp[c] = i
        
        res = []
        size = end = 0
        for i, c in enumerate(s):
            # keep updating size and end until it converges
            size += 1
            end = max(end, mp[c])
            # when i catches up to end, add to res
            if i == end:
                res.append(size)
                size = 0
                
        return res
        

    def checkValidString(self, s: str) -> bool:
        """
        https://neetcode.io/problems/valid-parenthesis-string
        Time: O(n), Space: O(n)
        """
        # two stacks for left parentheses and stars
        left, star = [], []
        for i in range(len(s)):
            # add to each stack
            if s[i] == '(':
                left.append(i)
            elif s[i] == '*':
                star.append(i)
            else:
                # if left and star are empty, we can't match the right parenthesis
                if not left and not star:
                    return False
                # pop from left first, otherwise pop from star
                if left:
                    left.pop()
                else:
                    star.pop()
                    
        # for the remaining items, treat star as right parenthesis
        while left and star:
            # if left's index is greater, we can't close it
            if left.pop() > star.pop():
                return False

        # check if left is empty
        return not left

        # # Greedy approach, Space O(1)
        # # min/max number of unmatched '('
        # leftMin, leftMax = 0, 0 
        # for c in s:
        #     if c == "(":
        #         leftMin, leftMax = leftMin + 1, leftMax + 1
        #     elif c == ")":
        #         leftMin, leftMax = leftMin - 1, leftMax - 1
        #     else:
        #         leftMin, leftMax = leftMin - 1, leftMax + 1
        #     if leftMax < 0:
        #         return False
        #     if leftMin < 0:
        #         leftMin = 0
        # return leftMin == 0
        
