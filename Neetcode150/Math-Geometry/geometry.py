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


    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        """
        https://neetcode.io/problems/spiral-matrix
        Time: O(m * n), Space: O(1)
        """
        res = []
        left, right = 0, len(matrix[0])
        top, bottom = 0, len(matrix)
        while left < right and top < bottom:
            # left to right
            for i in range(left, right):
                res.append(matrix[top][i])
            # update top pointer immediately to move down, same for other three
            top += 1
            # top to bottom
            for i in range(top, bottom):
                res.append(matrix[i][right - 1])
            right -= 1
            # important check: if the pointers are already out of bounds
            # stop to avoid duplicate counts (when only a single row/column left)
            if not (left < right and top < bottom):
                break
            # right to left
            for i in range(right - 1, left - 1, -1):
                res.append(matrix[bottom - 1][i])
            bottom -= 1
            # bottom to top
            for i in range(bottom - 1, top - 1, -1):
                res.append(matrix[i][left])
            left += 1
          
        return res


    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        https://neetcode.io/problems/set-zeroes-in-matrix
        Time: O(m * n), Space: O(1)
        """
        n_rows, n_cols = len(matrix), len(matrix[0])
        # additional tracker for the first row
        rowZero = False
        # go through each cell
        # update tracker in the first row (for columns) and in the first column (for rows)
        for r in range(n_rows):
            for c in range(n_cols):
                if matrix[r][c] == 0:
                    # update tracker for this column
                    matrix[0][c] = 0
                    # update tracker for this row
                    if r > 0:
                        matrix[r][0] = 0
                    else:
                        # if it's the first row, use the additional tracker
                        rowZero = True

        # update cells to zero based on the tracker
        # so only need to start from the second row/col
        for r in range(1, n_rows):
            for c in range(1, n_cols):
                if matrix[r][0] == 0 or matrix[0][c] == 0:
                    matrix[r][c] = 0

        # update first column
        if matrix[0][0] == 0:
            for r in range(n_rows):
                matrix[r][0] = 0

        # update first row
        if rowZero:
            for c in range(n_cols):
                matrix[0][c] = 0   
