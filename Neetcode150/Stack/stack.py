"""
Problems for stacks
"""

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
