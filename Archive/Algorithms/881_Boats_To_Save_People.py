# https://leetcode.com/problems/boats-to-save-people/
class Solution:
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        people.sort()
        # lightest and heaviest
        i, j = 0, len(people) - 1
        ans = 0
        while i <= j:
            ans += 1
            # if the current lightest person can sit with the heaviest person, do so
            if people[i] + people[j] <= limit:
                i += 1
            # the current heaviest person is definitely taking a boat with or without another
            j -= 1
        return ans
