"""
Problems for Graphs
"""
from typing import List


class Solution:

    def orangesRotting(self, grid: List[List[int]]) -> int:
        """
        https://leetcode.com/problems/rotting-oranges
        Time: O(m * n), Space: O(m * n)
        """
        q = collections.deque()
        fresh = 0
        time = 0
      
        # traverse the grid and add rotten oranges to the queue
        # also keep track of the number of fresh oranges
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == 1:
                    fresh += 1
                if grid[r][c] == 2:
                    q.append((r, c))
                  
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        # keep searching until the queue is empty or no more fresh ones
        while q and fresh > 0:
            length = len(q)
            # go through every rotten orange in the queue and increment time at the end
            for i in range(len(q)):
                r, c = q.popleft()
                # check all four directions
                for dr, dc in directions:
                    row, col = r + dr, c + dc
                    # if there are fresh oranges, decrement the count and mark as rotten
                    if row in range(len(grid)) and col in range(len(grid[0])) and grid[row][col] == 1:
                        grid[row][col] = 2
                        q.append((row, col))
                        fresh -= 1
            time += 1
          
        return time if fresh == 0 else -1
      
