"""
Problems for Linked List
"""



# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
      

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
