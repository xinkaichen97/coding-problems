class Solution(object):
    def twoSum(self, nums, target):
        dic = {}
        for i, num in enumerate(nums):
            if target - num not in dic:
                dic[num] = i
            else:
                return [dic[target - num], i]
        
