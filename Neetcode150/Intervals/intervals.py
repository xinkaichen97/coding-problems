"""
Problems for Intervals
"""
from typing import List


class Interval(object):
    """
    Definition of Interval
    """
    def __init__(self, start, end):
        self.start = start
        self.end = end
        
        
class Solution:

    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        """
        https://neetcode.io/problems/insert-new-interval
        Time: O(n), Space: O(1)
        """
        res = []
        for i, interval in enumerate(intervals):
            # current interval is smaller
            if interval[1] < newInterval[0]:
                res.append(interval)
            # current interval is greater, directly add the remaining intervals and return
            elif interval[0] > newInterval[1]:
                res.append(newInterval)
                return res + intervals[i:]
            else:
                # overlapping, take the min of starts and the max of ends to form a new interval
                start = min(interval[0], newInterval[0])
                end = max(interval[1], newInterval[1])
                newInterval = [start, end]

        # need to add the new interval if we haven't returned
        res.append(newInterval)
        return res


    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        https://neetcode.io/problems/merge-intervals
        Time: O(nlogn), Space: O(n) for Python sorting
        """
        # sort by starts
        intervals.sort(key=lambda x:x[0])
        # add the first interval in the output list
        res = [intervals[0]]
        
        # start with the second interval
        for i in range(1, len(intervals)):
            # compare its start with the last appended interval's end to detect overlapping
            if res[-1][1] >= intervals[i][0]:
                # if overlapped, simply update the end in the output
                res[-1][1] = max(res[-1][1], intervals[i][1])
            else:
                res.append(intervals[i])
                
        return res
        
        
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

    
    def minMeetingRooms(self, intervals: List[Interval]) -> int:
        """
        https://neetcode.io/problems/meeting-schedule-ii
        Time: O(nlogn), Space: O(n)
        """
        # sort start times and end times respectively
        start = sorted([i.start for i in intervals])
        end = sorted([i.end for i in intervals])
        
        # use two pointers
        s, e = 0, 0
        res, count = 0, 0
        
        while s < len(intervals):
            # if a meeting starts, increment s and count
            if start[s] < end[e]:
                count += 1
                s += 1
            # if a meeting ends, increment e and decrement count
            else:
                count -= 1
                e += 1
            # calculate the maximum
            res = max(res, count)
        return res

    
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
      
