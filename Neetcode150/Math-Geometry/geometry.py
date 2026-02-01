"""
Problems for Geometry
"""
from typing import List


class Solution:
  
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        https://neetcode.io/problems/rotate-matrix
        Time: O(n^2), Space: O(1)
        """
        n = len(matrix)

        # reverse columns
        # reverse() is faster than the manual reverse
        matrix.reverse()
        # for i in range(n // 2):
        #     matrix[i], matrix[n - 1 - i] = matrix[n - 1 - i], matrix[i]

        # transpose
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        
        # alternatively, transpose first and then reverse rows
        # for row in matrix:
        #     row.reverse()
      
