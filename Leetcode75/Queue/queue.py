"""
Problems for Queues
"""
from collections import deque


class RecentCounter:
    """
    https://leetcode.com/problems/number-of-recent-calls/
    Time: O(1) (per operation), Space: O(N)
    """

    def __init__(self):
        self.queue = deque()
    
    def ping(self, t: int) -> int:
        self.queue.append(t)
        # remove the calls outside of the window
        while self.queue[0] < t - 3000:
            queue.popleft()
        return len(self.queue)


class Solution:
    """
    https://leetcode.com/problems/dota2-senate/
    Time: O(N), Space: O(N)
    """
    def predictPartyVictory(self, senate: str) -> str:
        # two queues to store Radiants and Dires
        r, d = deque(), deque()
        n = len(senate)
        # add the indices to each queue
        for i, s in enumerate(senate):
            if s == 'R':
                r.append(i)
            else:
                d.append(i)
        # loop until one queue is empty
        while r and d:
            # the next senators removed from queues
            r_idx = r.popleft()
            d_idx = d.popleft()
            # if one has a lower index, it can ban its opponent
            # the opponent gets removed, the winner gets added to the end of the queue
            if r_idx < d_idx:
                r.append(r_idx + n)
            else:
                d.append(d_idx + n)
        # the party that remains wins
        return "Radiant" if r else "Dire"
      
