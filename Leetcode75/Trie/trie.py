"""
Problems for Tries (Prefix Trees)
"""


class TrieNode:
    # keep track of children and whether it's the end of the word
    def __init__(self):
        self.children = {}
        self.endOfWord = False
        self.words = [] # used in Search Suggestions System

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

    def searchPrefix(self, prefix: str) -> bool:
        cur = self.root
        for c in prefix:
            if c not in cur.children:
                return []
            cur = cur.children[c]
        return cur.words


class Solution:
    
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        """
        https://leetcode.com/problems/search-suggestions-system
        Time: O(m), Space: O(n), m - total chars in products, n - # of nodes in the Trie
        """
        # sort the array to make sure they are in lexical order
        products.sort()
        trie = Trie()
        res = []
        # insert word in the Trie
        for product in products:
            trie.insert(product)
            
        prefix = ""
        # add every char to the prefix and search for the top-3 words
        for ch in searchWord:
            prefix += ch
            words = trie.searchPrefix(prefix)
            # we can also update the insert function to keep max = 3
            res.append(words[:3])
            
        return res
        
