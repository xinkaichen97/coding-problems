# https://leetcode.com/problems/merge-sorted-array/
class Solution:
    def merge(self, nums1, m, nums2, n):
        p1, p2 = m - 1, n - 1
        # go from end to beginning so that the first elements of nums1 (if smaller) don't have to be changed
        for i in range(m + n - 1, -1, -1):
            if p1 < 0 or p2 < 0:
                break
            if nums1[p1] > nums2[p2]:
                nums1[i] = nums1[p1]
                p1 -= 1
            else:
                nums1[i] = nums2[p2]
                p2 -= 1
        # only need to update nums2 if it still have elementes left
        while p2 >= 0:
            nums1[i] = nums2[p2]
            i -= 1
            p2 -= 1
