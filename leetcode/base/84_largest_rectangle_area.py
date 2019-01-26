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
        # 57.97%
        # 维持一个上升的栈，栈用于存索引
        res = 0
        index = []
        len_h = len(heights)
        for i in range(len_h):
            # 若 栈不为空 且 栈顶元素 >= 当前元素，则 height=栈顶元素， weight=i-栈第2个元素+1
            while (len(index) != 0) and (heights[index[-1]] >= heights[i]):
                cur = index[-1]
                index.pop()
                res = max(res, heights[cur] * (i if len(index) == 0 else i - index[-1] - 1))
            # 维持一个上升的栈
            index.append(i)
        # 遍历完后，再执行一遍 index
        while len(index) != 0:
            cur = index[-1]
            index.pop()
            res = max(res, heights[cur] * (len_h if len(index) == 0 else len_h - index[-1] - 1))
        return res

    def largestRectangleArea2(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        # 100.00%
        # 使用单调栈的解法，栈用于存数值
        if not heights:
            return 0
        stack1 = [heights[0]]
        result = heights[0]
        len_h = len(heights)
        if len_h == 1:
            return result
        for h in heights[1:]:
            if h >= stack1[-1]:
                stack1.append(h)
            else:
                num = 1
                while stack1 and stack1[-1] > h:
                    value = stack1.pop()
                    result = max(result, value * num)
                    num += 1
                # 添加栈中的最小元素
                # while num >= 1:
                #     stack1.append(h)
                #     num -= 1
                stack1.extend([h] * num)
        size = len(stack1)
        for i in range(size):
            result = max(result, (size - i) * stack1[i])
        return result


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
    # res = Solution().largestRectangleArea(heights)
    res = Solution().largestRectangleArea2(heights)
    print(res)


if __name__ == '__main__':
    main()
