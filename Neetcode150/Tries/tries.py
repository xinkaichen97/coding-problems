"""
Problems for Tries
"""
from typing import List


class TrieNode:
    def __init__(self):
        self.children = {}
        self.endOfWord = False

    def addWord(self, word):
        cur = self
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endOfWord = True


class PrefixTree:
    """
    https://neetcode.io/problems/implement-prefix-tree
    Time: O(n), Space: O(t), n - len(word), t - total TrieNodes
    """

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        # start from the root and go through each char
        cur = self.root
        for c in word:
            # if character not found in children, append a new node
            if c not in cur.children:
                cur.children[c] = TrieNode()
            # go to the branch with c
            cur = cur.children[c]
        # important: set EOW to True when insert is done
        cur.endOfWord = True

    def search(self, word: str) -> bool:
        cur = self.root
        for c in word:
            # if character not found in children, immediately return False
            if c not in cur.children:
                return False
            cur = cur.children[c]
        # return EOW instead of True: it has to be an exact match
        return cur.endOfWord

    def startsWith(self, prefix: str) -> bool:
        cur = self.root
        for c in prefix:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return True


class WordDictionary:
    """
    https://neetcode.io/problems/design-word-search-data-structure
    Time: O(n), Space: O(t + n), n - len(word), t - total TrieNodes
    """
    
    def __init__(self):
        self.root = TrieNode()        

    def addWord(self, word: str) -> None:
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endOfWord = True

    def search(self, word: str) -> bool:
        # use dfs to find all matches with "."
        def dfs(i, root):
            cur = root
            # go through all indices
            for j in range(i, len(word)):
                c = word[j]
                if c == ".":
                    # find matches in all children
                    for child in cur.children.values():
                        if dfs(j + 1, child):
                            return True
                    return False
                else:
                    # check if c in children
                    if c not in cur.children:
                        return False
                    cur = cur.children[c]
                    
            return cur.endOfWord

        # run dfs from the start
        return dfs(0, self.root)


class Solution:
    
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        """
        https://neetcode.io/problems/search-for-word-ii
        Time: O(m * n * 4 * 3^(t-1) + s), Space: O(s)
        t - max len of words, s - sum of len of words
        At the first step, there are 4 options, and then at each level, there are 3 options (can't go back)
        """
        # initialize Trie and add all words
        root = TrieNode()
        for word in words:
            root.addWord(word)

        # define res and visit as sets
        n_rows, n_cols = len(board), len(board[0])
        res, visit = set(), set()

        # backtracking function
        def dfs(r, c, node, word):
            # return if out of bounds, or the current char is not in the current path
            # or the current char is already visited
            if r < 0 or c < 0 or r >= n_rows or c >= n_cols or board[r][c] not in node.children or (r, c) in visit:
                return

            # add the current index to visit
            visit.add((r, c))
            # go to the next node in the Trie
            node = node.children[board[r][c]]
            # add to the word and check if it's in the list (exists in the Tries)
            word += board[r][c]
            if node.endOfWord:
                res.add(word)

            # run backtracking function on four directions
            dfs(r + 1, c, node, word)
            dfs(r - 1, c, node, word)
            dfs(r, c + 1, node, word)
            dfs(r, c - 1, node, word)

            # remove the current index to backtrack
            visit.remove((r, c))

        # run backtracking for every cell
        for r in range(n_rows):
            for c in range(n_cols):
                dfs(r, c, root, "")
        
        return list(res)
        
