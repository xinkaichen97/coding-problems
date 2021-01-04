# https://leetcode.com/problems/merge-two-sorted-lists/

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        node_return = node = ListNode(-1)
        while l1 and l2:
            if l1.val < l2.val:
                node.next = ListNode(l1.val)                
                l1 = l1.next
            else:
                node.next = ListNode(l2.val)
                l2 = l2.next
            node = node.next
        if l1:
            node.next = l1
        else:
            node.next = l2
        return node_return.next
