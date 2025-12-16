"""
Problems for Graphs
"""
from typing import List
from collections import defaultdict


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
      

    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        """
        https://leetcode.com/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero
        Time: O(n), Space: O(n)
        """
        # define a set with the root node 
        seen = {0}
        # define graph and roads
        graph = defaultdict(list)
        roads = set()

        # DFS function
        def dfs(node):
            res = 0
            # for all the connections that the node has
            for neighbor in graph[node]:
                # if the neighbor is not yet seen
                if neighbor not in seen:
                    # if the connection is already in roads, need to reorder
                    if (node, neighbor) in roads:
                        res += 1
                    seen.add(neighbor)
                    # recursive call to start with neighbor
                    res += dfs(neighbor)
            return res

        # add all edges to the graph and roads
        for x, y in connections:
            graph[x].append(y)
            graph[y].append(x)
            roads.add((x,y))

        # result is the dfs call from 0
        return dfs(0)


    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        """
        https://leetcode.com/problems/evaluate-division
        Time: O(n * m), n - len(equations), m - len(queries), Space: O(n)
        """
        # define graph as a dict of dict
        graph = defaultdict(defaultdict)

        # backtracking function
        def backtrack(curr, target, accum, seen):
            # add the current node to the set
            seen.add(curr)
            ans = -1.0
            neighbors = graph[curr]
            # if target is a neighbor, multiply it with the accumulative product
            if target in neighbors:
                ans = accum * neighbors[target]
            # otherwise search in the neighbor
            else:
                for neighbor, val in neighbors.items():
                    if neighbor in seen:
                        continue
                    ans = backtrack(neighbor, target, accum * val, seen)
                    if ans != -1.0:
                        break
            # remove the current node to backtrack
            seen.remove(curr)
            return ans

        # add numerator and denominator pairs to the graph
        for (num, denom), val in zip(equations, values):
            graph[num][denom] = val
            graph[denom][num] = 1 / val
        
        res = []
        for num, denom in queries:
            # if not both in graph, return -1
            if num not in graph or denom not in graph:
                ans = -1.0
            # if two nums are the same, return 1
            elif num == denom:
                ans = 1.0
            # otherwise, find the query pair in the backtracking function
            else:
                seen = set()
                ans = backtrack(num, denom, 1, seen)
            res.append(ans)

        return res
