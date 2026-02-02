"""
Problems for stacks
"""
from typing import List


class Solution:
    
    def isValid(self, s: str) -> bool:
        """
        https://neetcode.io/problems/validate-parentheses
        Time: O(n), Space: O(n)
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
        Time: O(n), Space: O(n)
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


    def evalRPN(self, tokens: List[str]) -> int:
        """
        https://neetcode.io/problems/evaluate-reverse-polish-notation
        Time: O(n), Space: O(n)
        """
        stack = []
        ops = {'+', '-', '*', '/'}
        for token in tokens:
            # convert numbers to integers and add to stack
            if token not in ops:
                stack.append(int(token))
            else:
                # the second number is popped first
                num2 = stack.pop()
                num1 = stack.pop()
                # use eval to calculate
                res = eval(f"{num1} {token} {num2}")
                # add the result back to stack
                stack.append(int(res))
        return stack[0]
        

    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        """
        https://neetcode.io/problems/car-fleet
        Time: O(nlogn), Space: O(n)
        """
        # we can also just use prevTime
        stack = []
        # sort cars by positions in descending order
        cars = [(pos, sp) for pos, sp in zip(position, speed)]
        cars.sort(reverse=True)

        # if the time is lower than or equal to the previous time
        # that means they will form a car fleet
        for pos, sp in cars:
            time = (target - pos) / sp
            # need to add the first car too
            if not stack or time > stack[-1]:
                stack.append(time)
                
        # the length of the stack is the answer
        return len(stack)

        # # alternative solution using prevTime
        # res = 1
        # prevTime = (target - cars[0][0]) / cars[0][1]
        # for pos, sp in cars[1:]:
        #     time = (target - pos) / sp
        #     if time > prevTime:
        #         res += 1
        #         prevTime = time
        # return res


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
