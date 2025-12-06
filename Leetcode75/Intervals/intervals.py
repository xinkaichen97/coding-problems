"""
Problems for Intervals
"""
from typing import List


class Solution:
  
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        """
        https://leetcode.com/problems/non-overlapping-intervals
        Time: O(nlogn), Space: O(n) for Python sorting
        """
        # sort by start times
        intervals.sort()
        res = 0
        # keep track of previous end time
        prev = intervals[0][1]
      
        for i in range(1, len(intervals)):
            # if current start is lower than previous end, there's an overlap
            # remove the interval and update the new previous end if the current end is smaller
            if intervals[i][0] < prev:
                res += 1
                prev = min(prev, intervals[i][1])
            # if no overlap, still update the previous end
            else:
                prev = intervals[i][1]
              
        return res
