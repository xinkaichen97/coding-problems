# https://leetcode.com/problems/add-two-numbers/

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    # first way: keep adding the nodes
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        result = ListNode(0)
        result_head = result
        carry = 0
        
        while l1 or l2 or carry:
            val1 = (l1.val if l1 else 0)
            val2 = (l2.val if l2 else 0)
            result.next = ListNode((carry + val1 + val2) % 10)
            carry = (carry + val1 + val2) // 10
            result = result.next
            l1 = (l1.next if l1 else None)
            l2 = (l2.next if l2 else None)
        return result_head.next
    
    # second way: convert to numbers, add, and convert back to ListNode (slow)
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        str_l1, str_l2 = '', ''
        while l1:
            str_l1 += str(l1.val)
            l1 = l1.next
        while l2:
            str_l2 += str(l2.val)
            l2 = l2.next
        total = int(str_l1[::-1]) + int(str_l2[::-1])
        # reverse the result in order to make it a linked list
        total = str(total)[::-1]
        result = ListNode(0)
        result_head = result
        for chr in total:
            result.next = ListNode(int(chr))
            result = result.next
        return result_head.next
