# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/10
"""  
Usage Of '7_reverse.py' : 
"""

import numpy as np


class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        sign = 1 if x >= 0 else -1
        res_v = sign * int(str(abs(x))[::-1])
        if (res_v >= -np.power(2, 31)) and (res_v <= np.power(2, 31) - 1):
            return res_v
        else:
            return 0


def get_test_instance(example=1):
    x = 123
    if example == 1:
        # "321"
        pass
    if example == 2:
        x = -123
        # -321
    if example == 3:
        x = 120
    return x


def main():
    x = get_test_instance(example=1)
    # x = get_test_instance(example=2)
    # x = get_test_instance(example=3)
    res = Solution().reverse(x)
    print(res)


if __name__ == '__main__':
    main()
