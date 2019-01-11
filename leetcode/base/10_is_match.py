# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/11
"""  
Usage Of '10_is_match.py' : 
"""

import re


class Solution(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        res = re.findall(p, s)
        if (len(res) > 0) and (len(res[0]) == len(s)):
            return True
        else:
            return False


def get_test_instance(example=1):
    s = "aa"
    p = "a"
    if example == 1:
        pass
    if example == 2:
        s = "aa"
        p = "a*"
    if example == 3:
        s = "ab"
        p = ".*"
    if example == 4:
        s = "aab"
        p = "c*a*b"
    if example == 5:
        s = "mississippi"
        p = "mis*is*p*."
    return s, p


def main():
    s, p = get_test_instance(example=1)
    # s, p = get_test_instance(example=2)
    # s, p = get_test_instance(example=3)
    # s, p = get_test_instance(example=4)
    # s, p = get_test_instance(example=5)
    res = Solution().isMatch(s, p)
    print(res)


if __name__ == '__main__':
    main()
