"""
Problems for Two Pointers
"""
from typing import List


class Solution:
    
    def isPalindrome(self, s: str) -> bool:
        """
        https://neetcode.io/problems/is-palindrome
        Time: O(n), Space: O(1)
        """
        l, r = 0, len(s) - 1
        while l < r:
            # skip non-alphanumerical characters
            while l < r and not s[l].isalnum():
                l += 1
            while r > l and not s[r].isalnum():
                r -= 1
            # compare s[l] and s[r] (case-insensitive)
            if s[l].lower() != s[r].lower():
                return False
            l += 1
            r -= 1
        return True

  
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        """
        https://neetcode.io/problems/two-integer-sum-ii
        Time: O(n), Space: O(1)
        """
        # two pointers at both ends as the array is non-decreasing
        l, r = 0, len(numbers) - 1
        while l < r:
            if numbers[l] + numbers[r] == target:
                return [l + 1, r + 1] # 1-indexed
            elif numbers[l] + numbers[r] < target:
                l += 1
            else:
                r -= 1
        return []


    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """
        https://neetcode.io/problems/three-integer-sum
        Time: O(n^2), Space: O(n)
        Timsort (Python's default sorting algorithm) uses O(n) auxiliary space in the worst case
        """
        res = []
        nums.sort()
        for i in range(len(nums) - 2):
            # the lowest number has to be negative
            if nums[i] > 0:
                break
            # skip duplicates in the first number
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            target = -nums[i]
            # two pointers
            j = i + 1
            k = len(nums) - 1
            while j < k:
                currsum = nums[j] + nums[k]
                if currsum == target:
                    res.append([nums[i], nums[j], nums[k]])
                    j += 1
                    k -= 1
                    # increase j to skip duplicates
                    while nums[j] == nums[j - 1] and j < k:
                        j += 1
                elif currsum > target:
                    k -= 1
                else:
                    j += 1
        return res

    
    def maxArea(self, heights: List[int]) -> int:
        """
        https://neetcode.io/problems/max-water-container
        Time: O(n), Space: O(1)
        """
        i, j = 0, len(heights) - 1
        res = 0
        while i < j:
            # calculate current area
            res = max(res, (j - i) * min(heights[i], heights[j]))
            if heights[i] <= heights[j]:
                i += 1
            else:
                j -= 1
        return res


    def trap(self, height: List[int]) -> int:
        """
        https://neetcode.io/problems/trapping-rain-water
        Time: O(n), Space: O(1)
        """
        l, r = 0, len(height) - 1
        # keep the maximum heights on left and right
        leftMax, rightMax = height[l], height[r]
        res = 0
        while l < r:
            # if leftmax is lower, increment l
            if leftMax < rightMax:
                l += 1
                leftMax = max(leftMax, height[l])
                # max minus current height is the storage at current point
                res += leftMax - height[l]
            else:
                r -= 1
                rightMax = max(rightMax, height[r])
                res += rightMax - height[r]
        return res
      
