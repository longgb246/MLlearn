# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/3/3
"""  
Usage Of 'fuxi.py' : 
"""
from __future__ import print_function


def twoSum(nums, target):
    res_dict = {}
    for i, each in enumerate(nums):
        if res_dict.get(target - each) is not None:
            return [res_dict[target - each], i]
        res_dict[each] = i
    return -1


print('\n[ twoSum ]')
print(twoSum([2, 7, 11, 15], 9))


def lengthOfLongestSubstring(s):
    left = 0
    res = 0
    res_dict = {}
    for i, each in enumerate(s):
        if res_dict.has_key(each):
            left = max(res_dict.get(each) + 1, left)
        res_dict[each] = i
        res = max(i - left + 1, res)
    return res


print('\n[ lengthOfLongestSubstring ]')
print(lengthOfLongestSubstring('abcabcbb'))
print(lengthOfLongestSubstring('bbbbb'))
print(lengthOfLongestSubstring('pwwkew'))

import numpy as np


def longestPalindrome(s):
    s_join = '#' + '#'.join(list(s)) + '#'
    res = [0]
    for i in range(1, len(s_join)):
        res.append(0)
        for j in range(i)[::-1]:
            if (2 * i - j + 1) <= len(s_join) and (s_join[j] == s_join[2 * i - j]):
                res[i] += 1
            else:
                break
    max_i = int(np.argmax(res))
    s2 = s_join[(max_i - res[max_i]): (max_i + res[max_i] + 1)].replace('#', '')
    return s2


print('\n[ longestPalindrome ]')
print(longestPalindrome('babad'))
print(longestPalindrome('cbbd'))


class Tree():
    def MidOrder(self, node):
        print(node.value)
        if not node.left:
            self.MidOrder(node.left)
        if not node.right:
            self.MidOrder(node.right)
