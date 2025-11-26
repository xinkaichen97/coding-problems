"""
Problems for Binary Search
"""


class Solution:

    # The guess API is already defined for you.
    # def guess(num: int) -> int:
    # @param num, your guess
    # @return -1 if num is higher than the picked number
    #          1 if num is lower than the picked number
    #          otherwise return 0
    def guessNumber(self, n: int) -> int:
        """
        https://leetcode.com/problems/guess-number-higher-or-lower
        Time: O(logn), Space: O(1)
        """

        # initialize l, r as [0, n)
        l, r = 0, n
      
        # use <= because l can be the same as r
        while l <= r:
            mid = (l + r) // 2
            if guess(mid) == 0:
                return mid
            if guess(mid) == 1:
                l = mid + 1
            else:
                r = mid - 1


    def findPeakElement(self, nums: List[int]) -> int:
        """
        https://leetcode.com/problems/find-peak-element
        Time: O(logn), Space: O(1)
        """
        # initialize l, r as [0, n - 1]
        l, r = 0, len(nums) - 1
      
        # use < because we stop when l == r
        while l < r:
            mid = (l + r) // 2
            # if num at mid is greater than mid + 1, search in the left half
            if nums[mid] > nums[mid + 1]:
                r = mid
            # otherwise search in the left half
            else:
                l = mid + 1

        # return either l or r because they are the same
        return l


    def successfulPairs(self, spells: List[int], potions: List[int], success: int) -> List[int]:
        """
        https://leetcode.com/problems/successful-pairs-of-spells-and-potions
        Time: O((n+m)logm), Space: O(1)
        """
        res = []
        m = len(potions)
        
        # sort potions for binary search
        potions.sort()

        for spell in spells:
            idx = -1
            # binary search to find the index above which it's successful
            l, r = 0, m - 1
            while l <= r:
                mid = (l + r) // 2
                # if current product is successful, update index and right bound
                if spell * potions[mid] >= success:
                    idx = mid
                    r = mid - 1
                # otherwise update left bound
                else:
                    l = mid + 1
            # the number is total length - index, or zero if index is not updated
            res.append(m - idx if idx >= 0 else 0)

        return res


    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        """
        https://leetcode.com/problems/koko-eating-bananas
        Time: (nlogm), Space: O(1)
        n -> len(piles), m -> max(piles)
        """
        # helper function to see if Koko can finish at k
        def canFinish(k: int) -> bool:
            t = 0
            for p in piles:
                # if p is not divisible by k, add one (math.ceil)
                t += p // k
                if p % k:
                    t += 1
                if t > h:
                    return False
            return True

        # binary search from 1 to the max pile
        l, r = 1, max(piles)
        res = r
        while l <= r:
            mid = (l + r) // 2
            # if can finish, search in the left and update the answer
            if canFinish(mid):
                res = mid
                r = mid - 1
            # otherwise search in the right
            else:
                l = mid + 1
                
        return res
