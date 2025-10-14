# https://leetcode.com/problems/zigzag-conversion/

class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1:
            return s
        # result for each row
        rows = ["" for i in range(min(numRows, len(s)))]
        current_row = 0
        down = False
        for char in s:
            # add to current row
            rows[current_row] += char
            # change direction if at the top or bottom
            if current_row == 0 or current_row == numRows - 1:
                down = not down
            # either going down or up
            if down:
                current_row += 1
            else:
                current_row -= 1
        return "".join(rows)
