# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2018/12/29
"""  
Usage Of '2_add_two_number' : 
"""


# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


# ---------------------------------------
# 1、低效率 solution
def get_cal_value(l):
    if hasattr(l, 'val'):
        cal_v = l.val
    else:
        cal_v = l
    return cal_v


def get_next_node(l):
    if (hasattr(l, 'next')) and (l.next is not None):
        return l.next, True
    else:
        return 0, False


def cir_nodes(last_node, l1, l2, add_v):
    # print('l1 : ', get_cal_value(l1))
    # print('l2 : ', get_cal_value(l2))
    cal_v = get_cal_value(l1) + get_cal_value(l2) + add_v
    add_v, this_v = divmod(cal_v, 10)
    l1, l1_flag = get_next_node(l1)
    l2, l2_flag = get_next_node(l2)
    this_node = ListNode(this_v)
    last_node.next = this_node
    flag_add = any([l1_flag, l2_flag, add_v > 0])
    if flag_add:
        cir_nodes(this_node, l1, l2, add_v)


# ---------------------------------------
# Solution
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        add_v = 0
        res_node = ListNode(0)
        cir_nodes(res_node, l1, l2, add_v)
        return res_node.next


# ---------------------------------------
# Test Cases
def print_solution(solution):
    if hasattr(solution, 'val'):
        print solution.val,
    if (hasattr(solution, 'next')) and solution.next is not None:
        print ' -> ',
        print_solution(solution.next)


def get_test_instance1():
    l1_1 = ListNode(2)
    l1_2 = ListNode(4)
    l1_3 = ListNode(3)
    l1_2.next = l1_3
    l1_1.next = l1_2

    l2_1 = ListNode(5)
    l2_2 = ListNode(6)
    l2_3 = ListNode(4)
    l2_2.next = l2_3
    l2_1.next = l2_2
    return l1_1, l2_1


def get_test_instance2():
    l1_1 = ListNode(5)

    l2_1 = ListNode(5)
    return l1_1, l2_1


def main():
    l1, l2 = get_test_instance2()
    res = Solution().addTwoNumbers(l1, l2)
    # print_solution(l1)
    # print_solution(l2)
    print_solution(res)


if __name__ == '__main__':
    main()
