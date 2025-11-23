"""
Problems for Heap / Priority Queue
"""
import heapq
from typing import List


class Solution:
  
    def findKthLargest(self, nums: List[int], k: int) -> int:
        """
        https://leetcode.com/problems/kth-largest-element-in-an-array
        Time: O(nlogk), Space: O(k)
        """
        # create a heap with first k elements
        heap = nums[:k]
        heapq.heapify(heap)

        # check the remaining elements
        for num in nums[k:]:
            # keep replacing the smallest element (top of the heap)
            if num > heap[0]:
                # remove the smallest one and add the num to the heap
                heapq.heappop(heap)
                heapq.heappush(heap, num)

        # the heap is now the largest k elements, return the top 
        return heap[0]


class SmallestInfiniteSet:
"""
https://leetcode.com/problems/smallest-number-in-infinite-set
Time: O(logn), Space: O(logn)
"""

    def __init__(self):
        self.min_num = 1 # current minimum number
        self.heap = [] # a list to be converted to heap
        self.seen = set() # a set to keep track of items in heap

    def popSmallest(self) -> int:
        # if the heap is not empty, there are smaller numbers added back
        if self.heap:
            # remove from both the heap and the set
            num = heapq.heappop(self.heap)
            self.seen.remove(num)
        else:
            # pop the smallest number and increase min_num
            num = self.min_num
            self.min_num += 1
        return num

    def addBack(self, num: int) -> None:
        # only adds back if lower than the current minimum
        if self.min_num > num and num not in self.heap:
            # add to both the heap and the set
            heapq.heappush(self.heap, num)
            self.seen.add(num)


    def totalCost(self, costs: List[int], k: int, candidates: int) -> int:
        """
        https://leetcode.com/problems/total-cost-to-hire-k-workers
        Time: O((k+m)⋅logm), Space: O(m), m = candidates
        """
        # keep two min heaps for first and last candidates
        heap_left = costs[:candidates]
        heap_right = costs[max(candidates, len(costs) - candidates):]
        heapq.heapify(heap_left)
        heapq.heapify(heap_right)

        # keep the indices for the next candidate
        next_left, next_right = candidates, len(costs) - candidates - 1
        res = 0

        # k rounds
        for _ in range(k):
            # if the right heap is null or the left min is smaller
            if (not heap_right) or (heap_left and heap_left[0] <= heap_right[0]):
                # pop from the left and add to the final cost
                res += heapq.heappop(heap_left)
                # only add to the heap if there's a valid candidate to add
                if next_left <= next_right:
                    heapq.heappush(heap_left, costs[next_left])
                    next_left += 1
            # otherwise operate on the right heap
            else:
                res += heapq.heappop(heap_right)
                if next_left <= next_right:
                    heapq.heappush(heap_right, costs[next_right])
                    next_right -= 1
                  
        return res


    def maxScore(self, nums1: List[int], nums2: List[int], k: int) -> int:
        """
        https://leetcode.com/problems/maximum-subsequence-score
        Time: O(nlogn), Space: O(k)
        """
        # use a prefix sum to keep track of k items in nums1
        res, currsum = 0, 0
        # use a minheap to get the smallest num in current calculation
        heap = []
      
        # get nums1/nums2 combination and reverse sort by the nums2 values
        nums = sorted(zip(nums1, nums2), key=lambda x: -x[1])
      
        # start from the largest value in nums2
        for n1, n2 in nums:
            # add n1 to current sum and push to heap
            currsum += n1
            heapq.heappush(heap, n1)
          
            # if there are k items
            if len(heap) == k:
                # calculate the current score and compare with res
                res = max(res, currsum * n2)
                # remove the smallest num in heap to achieve maximum
                currsum -= heapq.heappop(heap)
              
        return res
      
