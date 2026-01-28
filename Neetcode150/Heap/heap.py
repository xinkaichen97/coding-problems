"""
Problems for Heaps / Priority Queues
"""
import heapq


class MedianFinder:
    """
    https://neetcode.io/problems/find-median-in-a-data-stream
    Time: addNum - O(m * logn), findMedian - O(m), Space: O(n)
    m - # of function calls, n - len(array)
    """
    # maintain two lists to be converted into heaps,
    # small - Max Heap (negate each item), large - Min Heap
    # push and pop - both O(logn)
    def __init__(self):
        self.small = []
        self.large = []

    def addNum(self, num: int) -> None:
        # if num is greater than the min of large, push to large
        if self.large and num > self.large[0]:
            heapq.heappush(self.large, num)
        # otherwise push to small (times -1 to maintain a max heap)
        else:
            heapq.heappush(self.small, -1 * num)

        # rebalance heaps
        # if large has too many elements, pop the smallest and push to small
        if len(self.large) > len(self.small) + 1:
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -1 * val)
        # if small has too many elements, pop the largest and push to large
        if len(self.small) > len(self.large) + 1:
            val = -1 * heapq.heappop(self.small)
            heapq.heappush(self.large, val)

    def findMedian(self) -> float:
        # if odd length and large has more, the median is the smallest in large
        if len(self.large) > len(self.small):
            return self.large[0]
        # if small has more, the median is the largest in small
        if len(self.small) > len(self.large):
            return -1 * self.small[0]
        # even length, calculate the mean
        return (-1 * self.small[0] + self.large[0]) / 2.0
        
        
