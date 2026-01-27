"""
Problems for Graphs
"""
from typing import List


# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
      

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
        
