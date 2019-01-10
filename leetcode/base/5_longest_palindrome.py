# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/9
"""  
Usage Of '5_longest_palindrome.py' : 
"""

import numpy as np


def get_all_str(s):
    s_all = '#'.join(list(s))
    return '#' + s_all + '#'


class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        s_all = get_all_str(s)
        all_len = len(s_all)
        p = []
        for i, t_s in enumerate(s_all):
            p.append(0)
            for j in range(i)[::-1]:
                if ((2 * i - j) < all_len) and (s_all[j] == s_all[2 * i - j]):
                    p[i] += 1
                else:
                    break
        max_i = int(np.argmax(p))
        res = s_all[(max_i - p[max_i]): (max_i + p[max_i] + 1)].replace('#', '')
        return res


def get_test_instance(example=1):
    str1 = "babad"
    # s = "babad"
    if example == 1:
        # "bab"
        # "aba"
        pass
    if example == 2:
        str1 = "cbbd"
        # "bb"
    return str1


def main():
    str1 = get_test_instance(example=1)
    # str1 = get_test_instance(example=2)
    res = Solution().longestPalindrome(str1)
    print(res)


if __name__ == '__main__':
    main()
