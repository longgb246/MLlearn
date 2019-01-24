# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/23
"""  
Usage Of '70_climb_stairs.py' : 
"""


class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 98.15%
        p = [1, 1, 2]
        i = 0
        if n <= 2:
            return n
        for j in range(3, n + 1):
            i = j % 3
            p[i] = p[i - 1] + p[i - 2]
        res = p[i]
        return res


def get_test_instance(example=1):
    n = 2
    if example == 1:
        pass
    if example == 2:
        n = 3
    if example == 3:
        n = 4
    return n


def main():
    n = get_test_instance(example=1)
    # n = get_test_instance(example=2)
    # n = get_test_instance(example=3)
    res = Solution().climbStairs(n)
    print(res)


if __name__ == '__main__':
    main()
