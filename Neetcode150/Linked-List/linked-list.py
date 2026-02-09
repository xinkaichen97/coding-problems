"""
Problems for Linked List
"""
from typing import Optional


# Definition for singly-linked list nodes
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# Definition for a list node with a random pointer.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random


class Solution:

    def reverseList(self, head: ListNode) -> ListNode:
        """
        https://neetcode.io/problems/reverse-a-linked-list
        Time: O(n), Space: O(1)
        """
        prev, curr = None, head
        while curr:
            # save next node to temp
            temp = curr.next
            # next node connects to prev
            curr.next = prev
            # prev goes to next (curr)
            prev = curr
            # curr is the previous next node
            curr = temp
        return prev


    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        """
        https://neetcode.io/problems/merge-two-sorted-linked-lists
        Time: O(m+n), Space: O(1)
        """
        head = curr = ListNode()
        while list1 and list2:
            if list1.val < list2.val:
                curr.next = list1
                list1 = list1.next
            else:
                curr.next = list2
                list2 = list2.next
            curr = curr.next
        # append the remaining nodes
        curr.next = list1 or list2
        return head.next


    def hasCycle(self, head: Optional[ListNode]) -> bool:
        """
        https://neetcode.io/problems/linked-list-cycle-detection
        Time: O(n), Space: O(1)
        """
        # fast and slow pointers
        fast, slow = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if fast == slow:
                return True
        return False
      

    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        https://neetcode.io/problems/reorder-linked-list
        Time: O(n), Space: O(1)
        """
        slow, fast = head, head.next
        # find mid point
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        # reverse second half
        second = slow.next
        prev = None
        slow.next = None
        while second:
            tmp = second.next
            second.next = prev
            prev = second
            second = tmp
        # merge two halves
        first, second = head, prev
        while second:
            tmp1 = first.next
            tmp2 = second.next
            first.next = second
            second.next = tmp1
            first = tmp1
            second = tmp2


    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """
        https://neetcode.io/problems/remove-node-from-end-of-linked-list
        Time: O(n), Space: O(1)
        """
        # dummy node before head
        dummy = ListNode(0, head)
        left = dummy
        right = head
        # right moves n times
        while n > 0:
            right = right.next
            n -= 1
        # move both left and right
        while right:
            left = left.next
            right = right.next
        # delete left.next
        left.next = left.next.next
        # head could be the node to delete, so cannot return head
        return dummy.next

    
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        """
        https://neetcode.io/problems/copy-linked-list-with-random-pointer
        Time: O(n), Space: O(n)
        """
        # two-pass solution with a hashmap
        # old to copy mapping, need to map None
        mapping = {None: None}

        # first pass, get old to copy mapping
        curr = head
        while curr:
            copy = Node(curr.val)
            mapping[curr] = copy
            curr = curr.next

        # second pass, for each copy node, get its next and random
        curr = head
        while curr:
            copy = mapping[curr]
            copy.next = mapping[curr.next]
            copy.random = mapping[curr.random]
            curr = curr.next

        # # one-pass solution with a hashmap
        # mapping = collections.defaultdict(lambda: Node(0))
        # mapping[None] = None
        # curr = head
        # while curr:
        #     mapping[curr].val = curr.val
        #     mapping[curr].next = mapping[curr.next]
        #     mapping[curr].random = mapping[curr.random]
        #     curr = curr.next
            
        return mapping[head]
      
    
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """
        https://neetcode.io/problems/add-two-numbers
        Time: O(m+n), Space: O(1)
        """
        # create a dummy node to return and the current node
        curr = head = ListNode()
        carry = 0
        # keep going if l1 or l2 or carry still exists
        while l1 or l2 or carry:
            v1 = l1.val if l1 else 0
            v2 = l2.val if l2 else 0
            val = v1 + v2 + carry
            # get carry and value
            carry = val // 10
            val = val % 10
            # add a new node and go to next
            curr.next = ListNode(val)
            curr = curr.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        return head.next


    def findDuplicate(self, nums: List[int]) -> int:
        """
        https://neetcode.io/problems/find-duplicate-integer
        Time: O(n), Space: O(1)
        """
        # Floyd's algorithm: find the start of the loop in a linked list
        # first, find the intersection of slow and fast
        slow, fast = 0, 0
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break

        # next, start a second slow pointer
        # the intersection of both slow pointers is the answer
        slow2 = 0
        while True:
            slow = nums[slow]
            slow2 = nums[slow2]
            if slow == slow2:
                return slow

    
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        https://neetcode.io/problems/merge-k-sorted-linked-lists
        Time: O(nlogk), Space: O(k)
        """
        # edge case: return None if lists is not valid
        if not lists or len(lists) == 0:
            return None

        # instead of merging lists one by one, merge lists in twos -> logk instead of k
        while len(lists) > 1:
            merged = []
            for i in range(0, len(lists), 2):
                # make sure that index is not out of bound
                l1 = lists[i]
                l2 = lists[i + 1] if i + 1 < len(lists) else None
                # use the existing function
                merged.append(self.mergeTwoLists(l1, l2))
            # update lists with merged
            lists = merged
    
        return lists[0]


# Definition for double-linked lists used in LRU Cache
class Node:
    def __init__(self, key, val):
        self.key, self.val = key, val
        self.prev = self.next = None

class LRUCache:
    """
    https://neetcode.io/problems/lru-cache
    Time: O(1) for get() & put(), Space: O(n)
    """
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = {}
        self.left, self.right = Node(-1, -1), Node(-1, -1)
        # double link left and right nodes
        self.left.next = self.right
        self.right.prev = self.left

    def get(self, key: int) -> int:
        if key in self.cache:
            # remove and insert again to add to the rightmost position (most recently used)
            self.remove(self.cache[key])
            self.insert(self.cache[key])
            return self.cache[key].val # only returns value
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        # if the key exists, replace it with the new value
        if key in self.cache:
            self.remove(self.cache[key])
        self.cache[key] = Node(key, value)
        self.insert(self.cache[key])
        # if capacity is reached, remove the leftmost node (the one after self.left)
        if len(self.cache) > self.cap:
            lru = self.left.next
            self.remove(lru)
            del self.cache[lru.key]
    
    def remove(self, node: Node) -> None:
        # skip current node
        # prev's next is next, and next's prev is prev
        oldPrev, oldNext = node.prev, node.next
        oldPrev.next = oldNext
        oldNext.prev = oldPrev

    def insert(self, node: Node) -> None:
        # add current node to the right
        # prev's next and next's prev are both the new node
        oldPrev, oldNext = self.right.prev, self.right
        oldPrev.next = oldNext.prev = node
        # update node's prev and next
        node.next, node.prev = oldNext, oldPrev
        
