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
      

    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        """
        https://neetcode.io/problems/reconstruct-flight-path
        Time: O(E^2), Space: O(E)
        """
        # sort for the lexical order
        tickets.sort()
        adj = defaultdict(list)
        for src, dst in tickets:
            adj[src].append(dst)
        
        res = ["JFK"]
        def dfs(node):
            # return if all nodes are traversed
            if len(res) == len(tickets) + 1:
                return True
            # go back if the node doesn't have outgoing edges
            if node not in adj:
                return False

            # copy the current list as we need to modify the original
            temp = adj[node].copy()
            for i, nb in enumerate(temp):
                # remove the current node and add to res
                adj[node].pop(i)
                res.append(nb)
                # if we find a valid path from nb, return
                if dfs(nb):
                    return True
                # otherwise backtrack
                adj[node].insert(i, nb)
                res.pop()
        
        dfs("JFK")
        return res

        # # Hierholzer's Algorithm - Time: O(ElogE)
        # adj = defaultdict(list)
        # for src, dst in sorted(tickets)[::-1]:
        #     adj[src].append(dst)
        # res = []
        # # post-order in the reversed list
        # def dfs(node):
        #     while adj[node]:
        #         nb = adj[node].pop()
        #         dfs(nb)
        #     res.append(node)
        # dfs("JFK")
        # return res[::-1]

  
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        """
        https://neetcode.io/problems/min-cost-to-connect-points
        Time: O(n^2 * logn), Space: O(n^2)
        """
        # create the adjacency list using Manhattan distances to each point
        adj = defaultdict(list)
        for i in range(len(points)):
            xi, yi = points[i]
            for j in range(i + 1, len(points)):
                xj, yj = points[j]
                dist = abs(xi - xj) + abs(yi - yj)
                adj[i].append((dist, j))
                adj[j].append((dist, i))

        # Prim's Algorithm
        res = 0
        visit = set()
        # min-heap: (dist, node)
        heap = [(0, 0)]
        while len(visit) < len(points):
            # pop the node with the smallest distance
            dist, node = heapq.heappop(heap)
            if node in visit:
                continue
            # add distance and add node to visit
            res += dist
            visit.add(node)
            # add all neighbors if not in visit
            for nbDist, nb in adj[node]:
                if nb not in visit:
                    heapq.heappush(heap, (nbDist, nb))
        
        return res

        # # Kruskal's Algorithm: use Union-Find
        # edges = []
        # dsu = DSU(len(points))
        # for i in range(len(points)):
        #     xi, yi = points[i]
        #     for j in range(i + 1, len(points)):
        #         xj, yj = points[j]
        #         dist = abs(xi - xj) + abs(yi - yj)
        #         edges.append((dist, i, j))
        # edges.sort()
        # res = 0
        # for dist, u, v in edges:
        #     if dsu.union(u, v):
        #         res += dist
        # return res
      

    def swimInWater(self, grid: List[List[int]]) -> int:
        """
        https://neetcode.io/problems/swim-in-rising-water
        Time: O(n^2 * logn), Space: O(n^2)
        """
        n = len(grid)
        directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]

        # use a min-heap (max height, row, col)
        heap = [(grid[0][0], 0, 0)]
        visit = set()
        while heap:
            t, r, c = heapq.heappop(heap)
            # if reaches the bottom right, return t
            if r == n - 1 and c == n - 1:
                return t
            visit.add((r, c))
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in visit:
                    # add the max of the max height so far and the current cell
                    maxT = max(t, grid[nr][nc])
                    heapq.heappush(heap, (maxT, nr, nc))

        # # Kruskal's Algorithm
        # dsu = DSU(n * n)
        # positions = sorted((grid[r][c], r, c) for r in range(n) for c in range(n))
        # for t, r, c in positions:
        #     for dr, dc in directions:
        #         nr, nc = r + dr, c + dc
        #         if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] <= t:
        #             dsu.union(r * n + c, nr * n + nc)
        #     if dsu.find(0) == dsu.find(n * n - 1):
        #         return t
                  
  
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        """
        https://neetcode.io/problems/cheapest-flight-path
        Time: O(n + m * k), Space: O(n), m = len(flights)
        """
        # prices: distances from src
        prices = [float("inf")] * n
        prices[src] = 0

        # Bell-Ford algorithm
        for i in range(k + 1):
            # use a copy to avoid accessing the wrong level
            temp = prices.copy()
            for s, d, p in flights:
                # if the source is unreachable, skip
                # if the price is lower, update temp (only update prices after processing the entire level)
                if prices[s] != float("inf") and prices[s] + p < temp[d]:
                    temp[d] = prices[s] + p
            prices = temp

        # get the min distance to dst
        return prices[dst] if prices[dst] < float("inf") else -1

        # # deque solution, Time: O(n * k), Space; O(n + m)
        # prices = [float("inf")] * n
        # prices[src] = 0
        # adj = [[] for _ in range(n)]
        # for u, v, cst in flights:
        #     adj[u].append([v, cst])

        # q = deque([(0, src, 0)])
        # while q:
        #     cst, node, stops = q.popleft()
        #     if stops > k:
        #         continue

        #     for nei, w in adj[node]:
        #         nextCost = cst + w
        #         if nextCost < prices[nei]:
        #             prices[nei] = nextCost
        #             q.append((nextCost, nei, stops + 1))

        # return prices[dst] if prices[dst] != float("inf") else -1
      
