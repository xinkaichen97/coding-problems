"""
Problems for Intervals
"""
from typing import List


class Solution:

    def canAttendMeetings(self, intervals: List[Interval]) -> bool:
        """
        https://neetcode.io/problems/meeting-schedule
        Time: O(nlogn), Space: O(n) for Python sorting
        """
        # sort by start times
        intervals.sort(key=lambda i: i.start)

        for i in range(1, len(intervals)):
            i1 = intervals[i - 1]
            i2 = intervals[i]

            # if previous end is greater than current start
            # there's a meeting conflict
            if i1.end > i2.start:
                return False
              
        return True

  
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        """
        https://neetcode.io/problems/non-overlapping-intervals
        Time: O(nlogn), Space: O(n)
        """
        # sort by end times
        intervals.sort(key = lambda x: x[1])

        # keep track of the previous end
        prevEnd = intervals[0][1]
        res = 0

        for i in range(1, len(intervals)):
            # if prev end is greater than start, there's an overlap
            # already sorted by end times so the current end must be greater, we remove the current interval
            if prevEnd > intervals[i][0]:
                res += 1
            # if no overlap, update the previous end with the current end
            else:
                prevEnd = intervals[i][1]

        return res
      
