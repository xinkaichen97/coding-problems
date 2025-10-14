"""
Problems for stacks
"""
from typing import List


class Solution:
    
    def isValid(self, s: str) -> bool:
      """
      https://neetcode.io/problems/validate-parentheses
      """
      stack = []  
      mapping = {']': '[', '}': '{', ')': '('}
      for ch in s:
        # if the second half is found, look for the first half in the stack
        if ch in mapping.keys():
          if stack and mapping[ch] = stack[-1]:
            stack.pop()
          else:
            # if the last item doesn't match, it's not valid
            return False
        else:
          stack.append(ch)
      return True if stack else False


    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        """
        https://neetcode.io/problems/daily-temperatures
        """
        stack = []
        res = []
        for i, t in enumerate(temperatures):
            # maintain a monotonic decreasing stack
            # if current temp is higher, pop from stack and update result
            while stack and t > stack[-1][0]:
                stackT, stackI = stack.pop()
                # index diff
                res[stackI] = i - stackI
            # (t, i) as the key since we need the index to update res
            stack.append((t, i))
        return res
        

class MinStack:
    """
    https://neetcode.io/problems/minimum-stack
    """
    # maintain another minStack that keeps the minimum for each addition
    def __init__(self):
        self.stack = []
        self.minStack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        # update minimum
        val = min(val, self.minStack[-1] if self.minStack else val)
        self.minStack.append(val)

    def pop(self) -> None:
        self.stack.pop()
        # need to remove from minStack too
        self.minStack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.minStack[-1]
