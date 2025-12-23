"""
Problems for Tries (Prefix Trees)
"""


class TrieNode:
    # keep track of children and whether it's the end of the word
    def __init__(self):
        self.children = {}
        self.endOfWord = False

class Trie:
    """
    https://leetcode.com/problems/implement-trie-prefix-tree
    Time: O(n), Space: O(t) for each function call, n - len(word), t - # of TrieNodes
    """
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        # start from the root
        cur = self.root
        # go through each character in the word
        for c in word:
            # if the next node doesn't exist, create a new node
            if c not in cur.children:
                cur.children[c] = TrieNode()
            # keep going to the next
            cur = cur.children[c]
        # mark end of word
        cur.endOfWord = True

    def search(self, word: str) -> bool:
        cur = self.root
        for c in word:
            # if the next character doesn't exist in trie, return False
            if c not in cur.children:
                return False
            cur = cur.children[c]
        # check if endOfWord is True
        return cur.endOfWord

    def startsWith(self, prefix: str) -> bool:
        cur = self.root
        for c in prefix:
            # if the next character doesn't exist in trie, return False
            if c not in cur.children:
                return False
            cur = cur.children[c]
        # no need to check endOfWord
        return True
