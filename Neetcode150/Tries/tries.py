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
            cur = cur.children
        return True
      
