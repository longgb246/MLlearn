# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/10
"""  
Usage Of 'my_atoi.py' : 
"""

import numpy as np


class Solution(object):
    def myAtoi(self, str1):
        """
        :type str1: str
        :rtype: int
        """
        str1 = str1.strip()
        len_str = len(str1)
        i = 0
        max_v = int(np.power(2, 31))
        sign = 1
        for i, s in enumerate(str1):
            if (i == 0) and (s == '-'):
                sign = -1
            elif (i == 0) and (s == '+'):
                sign = 1
            elif s.isdigit():
                pass
            else:
                i = i - 1
                break
        if (len_str == 0) or (i < 0) or ((i == 0) and (str1[0] == '-')) or ((i == 0) and (str1[0] == '+')):
            return 0
        # elif i > 10:
        #     if sign > 0:
        #         return max_v - 1
        #     else:
        #         return -max_v
        else:
            res = int(str1[:(i + 1)])
            if res < -max_v:
                return -max_v
            elif res > (max_v - 1):
                return max_v - 1
            else:
                return res


def get_test_instance(example=1):
    str1 = '42'
    if example == 1:
        pass
    if example == 2:
        str1 = '   -42'
    if example == 3:
        str1 = '4193 with words'
    if example == 4:
        str1 = 'words and 987'
    if example == 5:
        str1 = '-91283472332'
    if example == 6:
        str1 = "3.14159"
    if example == 7:
        str1 = "-+1"
    if example == 8:
        str1 = "  -0012a42"
    if example == 9:
        str1 = ""
    if example == 10:
        str1 = "+1"
    if example == 11:
        str1 = "+"
    if example == 12:
        str1 = "  0000000000012345678"
    return str1


def main():
    str1 = get_test_instance(example=1)
    # str1 = get_test_instance(example=2)
    # str1 = get_test_instance(example=3)
    # str1 = get_test_instance(example=4)
    # str1 = get_test_instance(example=5)
    # str1 = get_test_instance(example=6)
    # str1 = get_test_instance(example=7)
    # str1 = get_test_instance(example=8)
    # str1 = get_test_instance(example=9)
    # str1 = get_test_instance(example=10)
    # str1 = get_test_instance(example=11)
    # str1 = get_test_instance(example=12)
    res = Solution().myAtoi(str1)
    print(res)


if __name__ == '__main__':
    main()
