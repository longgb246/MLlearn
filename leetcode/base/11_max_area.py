# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/11
"""  
Usage Of '11_max_area.py' : 
"""


class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        # 较长线段的指针向内侧移动，矩形区域的面积将受限于较短的线段而不会获得任何增加
        # 移动指向较短线段的指针尽管造成了矩形宽度的减小，但却可能会有助于面积的增大
        max_v = 0
        l = 0
        r = len(height) - 1

        while l < r:
            max_v = max(max_v, min(height[l], height[r]) * (r - l))
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1

        res = max_v
        return res

    def maxArea1(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        # 该方法超时
        len_h = len(height)
        max_v = 0
        for i, v in enumerate(height):
            for j in range(i + 1, len_h):
                this_v = (j - i) * min(v, height[j])
                if this_v > max_v:
                    max_v = this_v
        res = max_v
        return res


def get_test_instance(example=1):
    height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
    if example == 1:
        pass
    return height


def main():
    height = get_test_instance(example=1)
    res = Solution().maxArea(height)
    print(res)


if __name__ == '__main__':
    main()
