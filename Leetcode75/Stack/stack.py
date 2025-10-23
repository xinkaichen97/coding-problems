"""
Problems for Stacks
"""


class Solution:
    def removeStars(self, s: str) -> str:
        """
        https://leetcode.com/problems/removing-stars-from-a-string/
        Time: O(N), Space: O(N)
        """
        stack = []
        for ch in s:
            # when seeing the star, remove the previous one
            if ch == "*":
                stack.pop()
            # add to stack if it's not a star
            else:
                stack.append(ch)
        return "".join(stack)


    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        """
        https://leetcode.com/problems/asteroid-collision/
        Time: O(N), Space: O(N)
        """
        stack = []
        for a in asteroids:
            # only collide when the current one is negative and the last one is positive
            while stack and a < 0 < stack[-1]:
                # if negative wins, remove the last one, and continue
                if stack[-1] + a < 0:
                    stack.pop()
                    continue
                # if equal, remove the last one and do not continue
                elif stack[-1] + a == 0: 
                    stack.pop()
                # if equal or positive, break without adding the current one
                break
            else:
                # current asteroid only gets added when the while condition is False
                stack.append(a)
        return stack


    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        """
        https://leetcode.com/problems/daily-temperatures/
        Time: O(N), Space: O(N)
        """
        res = [0] * len(temperatures)
        stack = []
        for i, t in enumerate(temperatures):
            # find the temperature lower than the current and its index
            while stack and t > stack[-1][0]:
                temp, idx = stack.pop()
                res[idx] = i - idx
            # use (t, i) as the key
            stack.append((t, i))
        return res
      
