"""
Problems for Graphs
"""
from typing import List


class Solution:
  
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        """
        https://leetcode.com/problems/keys-and-rooms
        
        """
        # use a mapping to check if a room is already opened
        seen = [False] * len(rooms)
        seen[0] = True

        # use a stack for DFS
        stack = [0]
        while stack:
            node = stack.pop()
            # for all the keys in the current room
            for key in rooms[node]:
                # if not seen, add to the stack
                if not seen[key]:
                    seen[key] = True
                    stack.append(key)
                  
        return all(seen)
