# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/17
"""  
Usage Of '32_longest_valid_parentheses.py' : 
"""

import numpy as np


class Solution(object):
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        len_s = len(s)
        if len_s == 0:
            return 0
        s_score = np.zeros(len_s)
        s_score[0] = 1 if s[0] == '(' else 0
        for i in range(1, len_s):
            if ((s[i] == '(') and (s[i - 1] == ')')) or ((s[i] == ')') and (s[i - 1] == '(')):
                s_score[i] = s_score[i - 1] + 1
            elif s[i] == '(':
                s_score[i] = 1
        max_len = int(s_score.max())
        res = max_len - 1 if divmod(max_len, 2)[1] == 1 else max_len
        return res


def get_test_instance(example=1):
    s = "(()"
    if example == 1:
        pass
    if example == 2:
        s = ")()())"
    if example == 3:
        s = ""
    if example == 4:
        s = "()(())"
    return s


def main():
    s = get_test_instance(example=1)
    # s = get_test_instance(example=2)
    # s = get_test_instance(example=3)
    # s = get_test_instance(example=4)
    # s = get_test_instance(example=5)
    res = Solution().longestValidParentheses(s)
    print(res)


if __name__ == '__main__':
    main()
