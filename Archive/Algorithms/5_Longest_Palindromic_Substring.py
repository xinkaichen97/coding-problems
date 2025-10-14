class Solution:
    # DP solution took too long to run
    # solution 1
    def longestPalindrome(self, s: str) -> str:
        if s == "":
            return s
        start, end = 0, 0
        for i in range(len(s)):
            len1 = self.expandFromCenter(s, i, i)  # center = i-th, for odd length
            len2 = self.expandFromCenter(s, i, i + 1)  # center = i-th and (i+1)-th, for even length
            l = max(len1, len2)
            if l > end - start:
                start = i - (l - 1) // 2  # because i is either the single center or the left half of the center
                end = i + l // 2
        return s[start: end + 1]
    
    # find the length of the longest palindrome, starting from center
    def expandFromCenter(self, s, left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1
