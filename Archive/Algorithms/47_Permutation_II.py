class Solution:
    # backtracking
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        results = []
        def explore(comb, counter):
            # base case
            if len(comb) == len(nums):
                results.append(list(comb))
                return
            for num in counter:
                if counter[num] > 0:
                    comb.append(num)
                    counter[num] -= 1
                    # recursion
                    explore(comb, counter)
                    # restore previous state
                    comb.pop()
                    counter[num] += 1
        explore([], Counter(nums))
        return results
