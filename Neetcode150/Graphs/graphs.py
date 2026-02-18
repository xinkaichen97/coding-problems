"""
Problems for Graphs
"""
from collections import deque
from typing import List


# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
      

class UnionFind:
    """
    UnionFind class for efficient graph operations
    Time: O(α(n)) -> O(1) for union() & find(), Space: O(n)
    """
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n
    
    def find(self, x):
        # path compression: point parent to the root
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        
        if rootX == rootY:
            return
        
        if self.rank[rootX] > self.rank[rootY]:
            self.parent[rootY] = rootX
        elif self.rank[rootX] < self.rank[rootY]:
            self.parent[rootX] = rootY
        else:
            # add rank only when heights are equal
            self.parent[rootY] = rootX
            self.rank[rootX] += 1
        
        self.count -= 1


class Solution:
  
    def numIslands(self, grid: List[List[str]]) -> int:
        """
        https://neetcode.io/problems/count-number-of-islands
        Time: O(m * n), Space: O(m * n)
        """
        # initialize directions
        directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        n_row, n_col = len(grid), len(grid[0])
        res = 0

        # dfs function
        def dfs(r, c):
            # invalid cases: out of bounds
            if r < 0 or c < 0 or r >= n_row or c >= n_col or grid[r][c] == "0":
                return

            # set current cell to 0 to avoid repeated search
            grid[r][c] = "0"
            # run dfs for all four directions
            for dr, dc in directions:
                dfs(r + dr, c + dc)

        # for each cell with 1, run dfs and update res
        for r in range(n_row):
            for c in range(n_col):
                if grid[r][c] == "1":
                    dfs(r, c)
                    res += 1
        
        return res


    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        """
        https://neetcode.io/problems/max-area-of-island
        Time: O(m * n), Space: O(m * n)
        """
        directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        m, n = len(grid), len(grid[0])

        # BFS
        def bfs(r, c):
            q = deque()
            # mark current cell as 0 and add to queue
            grid[r][c] = 0
            q.append((r, c))
            # assumes land so res starts with 1
            res = 1

            while q:
                r, c = q.popleft()
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    # skip if out of bounds or water
                    if nr < 0 or nc < 0 or nr >= m or nc >= n or grid[nr][nc] == 0:
                        continue
                    # add neighbor to queue and mark as 0
                    q.append((nr, nc))
                    grid[nr][nc] = 0
                    # update area when adding to queue, not when dequeuing
                    res += 1
            return res

        # # DFS alternative
        # def dfs(r, c):
        #     if r < 0 or c < 0 or r >= m or c >= n or grid[r][c] == 0:
        #         return 0
        #     grid[r][c] = 0
        #     res = 1
        #     for dr, dc in directions:
        #         res += dfs(r + dr, c + dc)
        #     return res
        
        res = 0
        for r in range(m):
            for c in range(n):
                # only check when it's land (so no need to check in BFS)
                if grid[r][c] == 1:
                    res = max(res, bfs(r, c))
        
        return res
        
    
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        """
        https://neetcode.io/problems/max-area-of-island
        Time: O(m * n), Space: O(m * n)
        """
        m, n = len(grid), len(grid[0])

        def dfs(r, c):
            # if out of bounds or not an island or already visited
            if r < 0 or c < 0 or r >= m or c >= n or grid[r][c] == 0:
                return 0
            # mark as visited
            grid[r][c] = 0
            # sum all directions and add 1
            return 1 + dfs(r + 1, c) + dfs(r - 1, c) + dfs(r, c + 1) + dfs(r, c - 1)

        # go through every grid and get max 
        res = 0
        for r in range(m):
            for c in range(n):
                res = max(res, dfs(r, c))
        
        return res

    
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        """
        https://neetcode.io/problems/clone-graph
        Time: O(V + E), Space: O(V), V - vertices, E - edges
        """
        # create a mapping from ori to copy
        mapping = {}
        if not node:
            return None

        # dfs function
        def dfs(node):
            # return the copy if already in mapping, so that it doesn't get added more than once
            if node in mapping:
                return mapping[node]

            # create a copy and add to mapping
            copy = Node(node.val)
            mapping[node] = copy
          
            # create copies for all neighbors and add to the copy's neighbors
            # the edges are already constructed so space complexity is O(v)
            for nb in node.neighbors:
                copy.neighbors.append(dfs(nb))
            return copy
        
        return dfs(node)


    def islandsAndTreasure(self, grid: List[List[int]]) -> None:
        """
        https://neetcode.io/problems/islands-and-treasure
        Time: O(m * n), Space: O(m * n)
        """
        m, n = len(grid), len(grid[0])
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]

        # multi-source BFS: add all treasures/gates to the queue
        q = deque()
        for r in range(m):
            for c in range(n):
                if grid[r][c] == 0:
                    q.append((r, c))

        # go through each level
        level = 1
        while q:
            for _ in range(len(q)):
                r, c = q.popleft()
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    # here we process only when in bounds and the cell is land
                    # so we don't need a visit set; we do if we can't modify the grid
                    if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 2147483647:
                        q.append((nr, nc))
                        grid[nr][nc] = level
            level += 1


    def orangesRotting(self, grid: List[List[int]]) -> int:
        """
        https://neetcode.io/problems/rotting-fruit
        Time: O(m * n), Space: O(m * n)
        """
        q = deque()
        fresh = 0
        time = 0
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]

        # count fresh oranges and add rotten oranges to queue
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == 1:
                    fresh += 1
                if grid[r][c] == 2:
                    q.append((r, c))

        # loop until all rotten oranges are processed or there are no more fresh ones
        while q and fresh > 0:
            length = len(q)
            for i in range(len(q)):
                r, c = q.popleft()
                for dr, dc in directions:
                    row, col = r + dr, c + dc
                    # directly modify grid, add adjacent fresh oranges to queue
                    if row in range(len(grid)) and col in range(len(grid[0])) and grid[row][col] == 1:
                        grid[row][col] = 2
                        q.append((row, col))
                        fresh -= 1
            time += 1
            
        return time if fresh == 0 else -1

    
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        """
        https://neetcode.io/problems/pacific-atlantic-water-flow
        Time: O(m * n), Space: O(m * n)
        """
        # initialize two sets for valid points in Pacific or Atlantic
        n_rows, n_cols = len(heights), len(heights[0])
        pac, atl = set(), set()

        # dfs function
        def dfs(r, c, visit, prev):
            # if already visited, or current height is lower than previous height (meaning the water can't flow from the current point)
            # or if out of bounds, simply return
            if (r, c) in visit or r < 0 or c < 0 or r >= n_rows or c >= n_cols or heights[r][c] < prev:
                return
            # current point is valid, add to visit
            visit.add((r, c))
            # perform dfs on all adjacent points
            dfs(r + 1, c, visit, heights[r][c])
            dfs(r - 1, c, visit, heights[r][c])
            dfs(r, c + 1, visit, heights[r][c])
            dfs(r, c - 1, visit, heights[r][c])

        # start dfs from the top row (can flow to Pacific) and the bottom row (Atlantic)
        for c in range(n_cols):
            dfs(0, c, pac, heights[0][c])
            dfs(n_rows - 1, c, atl, heights[n_rows - 1][c])

        # also run dfs from the leftmost column (pacific) to the rightmost column (atlantic)
        for r in range(n_rows):
            dfs(r, 0, pac, heights[r][0])
            dfs(r, n_cols - 1, atl, heights[r][n_cols - 1])

        # check if the point is in both sets
        res = []
        for r in range(n_rows):
            for c in range(n_cols):
                if (r, c) in pac and (r, c) in atl:
                    res.append([r, c])
        
        return res


    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """
        https://neetcode.io/problems/course-schedule
        Time: O(V + E), Space: O(V + E)
        """
        # create a dependency mapping
        mapping = defaultdict(list)
        for crs, pre in prerequisites:
            mapping[crs].append(pre)
        # use to visit set to detect cycles
        visit = set()

        # dfs function on a node
        def dfs(crs):
            # if in visit, immediately return False
            if crs in visit:
                return False
            # if the dependency is empty, immediately return True
            if mapping[crs] == []:
                return True
            # add the current node to visit and check all its dependencies
            visit.add(crs)
            for pre in mapping[crs]:
                if not dfs(pre):
                    return False
            # IMPORTANT: removing from visit is necessary because one node can be reached from different paths
            visit.remove(crs)
            # set dependencies to [] to avoid repeated work
            mapping[crs] = []
            return True

        # need to check all courses
        for crs in range(numCourses):
            if not dfs(crs):
                return False
                
        return True
        

    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        """
        https://neetcode.io/problems/valid-tree
        Time: O(V + E), Space: O(V + E)
        """
        # if the graph is not fully connected, immediately return false
        if len(edges) != n - 1:
            return False
        # create and update mapping, add BOTH directions (undirected graph)
        mapping = defaultdict(list)
        for src, dst in edges:
            mapping[src].append(dst)
            mapping[dst].append(src)
        
        visit = set()
        # need to keep track of the previous node to avoid false positives
        def dfs(node, prev):
            if node in visit:
                return False
                
            visit.add(node)
            for nb in mapping[node]:
                # since it's both directions, prev is in the neighbors, so skip
                if nb == prev:
                    continue
                if not dfs(nb, node):
                    return False
            return True

        # only need to run dfs once since it's supposed to be fully connected
        if not dfs(0, -1):
            return False

        # check if visit has all the nodes
        return len(visit) == n
        

    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        """
        https://neetcode.io/problems/count-connected-components
        Time: O(V + E * α(V)) ~ O(V + E), Space: O(V)
        """
        parent = list(range(n))
        rank = [1] * n

        # find function
        def find(node):
            res = node
            # path compression - keep going until finding the root (O(V) -> O(1))
            while res != parent[res]:
                # optimization: jump to grandparent
                parent[res] = parent[parent[res]]
                # update res to its parent (actually grandparent)
                res = parent[res]
            return res

        # union function, time complexity α(V), α - Inverse Ackermann Function
        def union(u, v):
            # find parents of u and v
            pu = find(u)
            pv = find(v)
            # if they share the same parent, do not decrease # of connected components
            if pu == pv:
                return 0
            # if not, compare the ranks and add to the node with the higher ranks
            if rank[pu] > rank[pv]:
                parent[pv] = pu
                rank[pu] += rank[pv]
            else:
                parent[pu] = pv
                rank[pv] += rank[pu]
            return 1

        # start with n disconnected components
        res = n
        # for each newly connected pair, decrease from res
        for u, v in edges:
            res -= union(u, v)
        return res
        
