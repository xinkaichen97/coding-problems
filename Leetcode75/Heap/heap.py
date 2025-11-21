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
      
