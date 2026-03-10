"""
Problems for Graphs
"""
from collections import defaultdict, deque
import heapq
from typing import List


class Solution:
  
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        """
        https://neetcode.io/problems/network-delay-time
        Time: O(E * logV), Space: O(V + E)
        """
        # update the adjacency list (directed)
        adj = defaultdict(list)
        for u, v, w in times:
            adj[u].append((v, w))
        
        visit = set()
        # min-heap for (time, node), starting from k
        # worst case in heap - E elements, O(logE) <= O(log(V^2)) = O(logV)
        heap = [(0, k)]
        res = 0
        while heap:
            time, node = heapq.heappop(heap)
            # if the node is already visited, skip because we have found a shorter path to it
            if node not in visit:
                visit.add(node)
                res = time
                # update time and add neighbors
                for nb, w in adj[node]:
                    heapq.heappush(heap, (time + w, nb))
                  
        # return -1 if not all nodes are visited
        return res if len(visit) == n else -1

        # # DFS solution - O(V * E) time
        # adj = defaultdict(list)
        # for u, v, w in times:
        #     adj[u].append((v, w))
        # dist = {node: float("inf") for node in range(1, n + 1)}
        # def dfs(node, time):
        #     if time >= dist[node]:
        #         return
        #     dist[node] = time
        #     for nb, w in adj[node]:
        #         dfs(nb, time + w)
        # dfs(k, 0)
        # res = max(dist.values())
        # return res if res < float("inf") else -1
      
