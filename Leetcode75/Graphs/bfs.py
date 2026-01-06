"""
Problems for Graphs
"""
from typing import List


class Solution:

    def nearestExit(self, maze: List[List[str]], entrance: List[int]) -> int:
        """
        https://leetcode.com/problems/nearest-exit-from-entrance-in-maze
        Time: O(m * n), Space: O(m + n)
        """
        rows, cols = len(maze), len(maze[0])
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]

        # mark entrance as a wall because it's not an exit
        start_row, start_col = entrance
        maze[start_row][start_col] = "+"

        # define queue and add the entrance point
        q = collections.deque()
        q.append([start_row, start_col, 0])

        # maintain the queue for BFS
        while q:
            # get current location and distance
            curr_row, curr_col, curr_distance = q.popleft()

            # check all four directions
            for dr, dc in directions:
                next_row = curr_row + dr
                next_col = curr_col + dc

                # if the move is valid and there's an empty cell
                if 0 <= next_row < rows and 0 <= next_col < cols and maze[next_row][next_col] == ".":
                    # if at the border, we found the exit, return the distance traveled
                    if 0 == next_row or next_row == rows - 1 or 0 == next_col or next_col == cols - 1:
                        return curr_distance + 1

                    # mark the next move as visited
                    maze[next_row][next_col] = "+"
                    q.append([next_row, next_col, curr_distance + 1])
        
        return -1

    
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
      
