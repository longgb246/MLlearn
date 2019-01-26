# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/24
"""  
Usage Of '85_maximal_rectangle.py' : 
"""


class Solution(object):
    def largestRectangleArea(self, heights):
        if not heights:
            return 0
        stack = [heights[0]]
        res = heights[0]
        if len(heights) == 1:
            return res
        for h in heights[1:]:
            if stack[-1] <= h:
                stack.append(h)
            else:
                width = 1
                while stack and stack[-1] > h:
                    height = stack.pop()
                    res = max(res, height * width)
                    width += 1
                stack.extend([h] * width)
        for i in range(len(stack)):
            res = max(res, stack[i] * (len(stack) - i))
        return res

    def maximalRectangle(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        # 144 ms - 34.95%
        # 使用最大面积的那道题来做的
        if not matrix or not matrix[0]:
            return 0
        n = len(matrix[0])
        heights = [0] * n
        res = 0
        for row in matrix:
            heights = [0 if v == '0' else heights[j] + 1 for j, v in enumerate(row)]
            res = max(res, self.largestRectangleArea(heights))
        return res

    def maximalRectangle2(self, matrix):
        # 99.03%
        if not matrix or not matrix[0]:
            return 0
        n = len(matrix[0])
        # 增加一个末尾的0，当运行到最后的时候也自动触发计算
        height = [0] * (n + 1)
        ans = 0
        for row in matrix:
            for i in xrange(n):
                height[i] = height[i] + 1 if row[i] == '1' else 0
            stack = [-1]  # 表示为stack为空的时候，索引到height的0
            for i in xrange(n + 1):
                while height[i] < height[stack[-1]]:
                    h = height[stack.pop()]
                    w = i - 1 - stack[-1]
                    ans = max(ans, h * w)
                stack.append(i)
        return ans


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
