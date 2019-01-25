# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/25
"""  
Usage Of '84_largest_rectangle_area.py' : 
"""


class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        len_h = len(heights)
        index = []
        area = 0
        for i in range(len_h):
            if (len(index) == 0) or (heights[index[-1]] < heights[i]):
                index.append(i)
            else:
                while (len(index) != 0) and (heights[index[-1]] >= heights[i]):
                    tmp = index[-1]
                    index.pop()
                    if len(index) == 0:
                        length = i
                    else:
                        length = i - index[-1] - 1
                    area = max(area, length * heights[tmp])
                index.append(i)
        while len(index) != 0:
            tmp = index[-1]
            index.pop()
            if len(index) == 0:
                length = len_h
            else:
                length = len_h - index[-1] - 1
            area = max(area, length * heights[tmp])
        return area


def get_test_instance(example=1):
    heights = [2, 1, 5, 6, 2, 3]
    if example == 1:
        pass
    if example == 2:
        heights = "intention"
    return heights


def main():
    heights = get_test_instance(example=1)
    # heights = get_test_instance(example=2)
    res = Solution().largestRectangleArea(heights)
    print(res)


if __name__ == '__main__':
    main()
