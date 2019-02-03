# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/27
"""  
Usage Of '470_rand10.py' : 
"""

from numpy import random


def rand7():
    return 1


class Solution(object):
    def rand10(self):
        """
        :rtype: int
        """
        # half = rand7()
        # while half > 5:
        #     half = rand7()
        # mul = rand7()
        # while mul == 3:
        #     mul = rand7()
        # res = half + 5 if mul > 3 else half
        # return res
        return random.randint(1, 10)


def get_test_instance(example=1):
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    if example == 1:
        pass
    if example == 2:
        heights = "intention"
    return matrix


def main():
    matrix = get_test_instance(example=1)
    # heights = get_test_instance(example=2)
    res = Solution().rand10()
    print(res)


if __name__ == '__main__':
    main()
