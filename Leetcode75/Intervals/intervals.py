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


    def findMinArrowShots(self, points: List[List[int]]) -> int:
        """
        https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons
        Time: O(nlogn), Space: O(n)
        """
        if not points:
            return 0

        # sort by end times
        points.sort(key=lambda x:x[1])
        res = 1
        curr_end = points[0][1]
      
        for start, end in points:
            # if the current end is lower than start, no overlap
            # need an additional arrow
            if curr_end < start:
                res += 1
                curr_end = end
              
        return res
      
