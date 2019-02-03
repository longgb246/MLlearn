# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/27
"""  
Usage Of '48_rotate.py' : 
"""


class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        # 28 ms - 99.49%
        for i, v in enumerate(zip(*matrix[::-1])):
            matrix[i] = v


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
    # res = Solution().largestRectangleArea(heights)
    res = Solution().rotate(matrix)
    print(res)


if __name__ == '__main__':
    main()
