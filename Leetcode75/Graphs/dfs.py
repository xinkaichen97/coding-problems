"""
Problems for Graphs
"""
from typing import List


class Solution:
  
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        """
        https://leetcode.com/problems/keys-and-rooms
        Time: O(n+k), k - len(keys), Space: O(n)
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


    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        """
        https://leetcode.com/problems/number-of-provinces
        Time: O(n^2), Space: O(n)
        """
        n = len(isConnected)
        res = 0
        visit = [False] * n

        # DFS function
        def dfs(node, isConnected, visit):
            # update visit
            visit[node] = True
            # check all nodes
            for i in range(n):
                # only calls dfs if the city is connected but not visited
                if isConnected[node][i] and not visit[i]:
                    dfs(i, isConnected, visit)

        # if one city is not yet visited, increment the answer
        for i in range(n):
            # recursive call to update visit
            if not visit[i]:
                res += 1
                dfs(i, isConnected, visit)
        
        return res
      
