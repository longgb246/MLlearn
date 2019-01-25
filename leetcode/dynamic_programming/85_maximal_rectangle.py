# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/24
"""  
Usage Of '85_maximal_rectangle.py' : 
"""


class Solution(object):
    def maximalRectangle(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        pass


def get_test_instance(example=1):
    matrix = [
        ["1", "0", "1", "0", "0"],
        ["1", "0", "1", "1", "1"],
        ["1", "1", "1", "1", "1"],
        ["1", "0", "0", "1", "0"]
    ]
    if example == 1:
        pass
    if example == 2:
        matrix = "intention"
    return matrix


def main():
    matrix = get_test_instance(example=1)
    # matrix = get_test_instance(example=2)
    res = Solution().maximalRectangle(matrix)
    print(res)


if __name__ == '__main__':
    main()
