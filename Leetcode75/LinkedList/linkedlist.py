"""
Problems for Linked Lists
"""
from typing import Optional


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
  
    def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        https://leetcode.com/problems/delete-the-middle-node-of-a-linked-list/
        Time: O(N), Space: O(1)
        """
        if not head:
            return None
        # define a prev node to return
        prev = ListNode(0)
        prev.next = head
        # slow and faster pointers
        slow, fast = prev, head
        # fast moves two nodes until the end
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        # slow is now at the node before the middle
        # skip the middle by pointing to its next
        slow.next = slow.next.next
        return prev.next


    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        https://leetcode.com/problems/odd-even-linked-list/
        Time: O(N), Space: O(1)
        """
        if not head:
            return None
        # create odd and even pointers
        odd, even = head, head.next
        evenHead = even
        # keep going until even reaches the end
        while even and even.next:
            # update odd sequence
            odd.next = odd.next.next
            odd = odd.next
            # update even sequence
            even.next = even.next.next
            even = even.next
        # connect both sequences
        odd.next = evenHead
        return head

  
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        https://leetcode.com/problems/reverse-linked-list/
        Time: O(N), Space: O(1)
        """
        prev = None
        curr = head
        while curr:
            # save curr.next
            temp = curr.next
            # curr points to prev
            curr.next = prev
            # prev moves ahead
            prev = curr
            # curr moves ahead
            curr = temp
        # return prev as curr is now None
        return prev


    def pairSum(self, head: Optional[ListNode]) -> int:
        """
        https://leetcode.com/problems/maximum-twin-sum-of-a-linked-list/
        Time: O(N), Space: O(1)
        """
        # find the mid point using slow and fast pointers
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        # reverse the second half
        curr, prev = slow, None
        while curr:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
    
        # add nodes from both halves and update the max
        first, second = head, prev
        res = 0
        while second:
            res = max(res, first.val + second.val)
            first = first.next
            second = second.next
          
        return res
      
