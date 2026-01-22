"""
Problems for Tries
"""


class TrieNode:
    def __init__(self):
        self.children = {}
        self.endOfWord = False


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
