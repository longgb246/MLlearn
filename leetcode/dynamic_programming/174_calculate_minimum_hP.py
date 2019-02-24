# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/2/5
"""  
Usage Of '174_calculate_minimum_hP.py' :
"""


class Solution(object):
    def calculateMinimumHP(self, dungeon):
        """
        :type dungeon: List[List[int]]
        :rtype: int
        """
        min_dp = []


def get_test_instance(example=1):
    dungeon = [[-2, -3, 3],
               [-5, -10, 1],
               [10, 30, -5]]
    if example == 1:
        pass
    return dungeon


def main():
    dungeon = get_test_instance(example=1)
    # dungeon = get_test_instance(example=2)
    res = Solution().calculateMinimumHP(dungeon)
    print(res)


if __name__ == '__main__':
    main()
