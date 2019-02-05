# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/2/3
"""  
Usage Of '152_max_product.py' : 
"""


class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 28 ms - 99.66%
        # 计算从左到右的相乘的最大值，和计算从右到左的最大值；再将两组最大值相比
        B = nums[::-1]
        for i in range(1, len(nums)):
            nums[i] *= nums[i - 1] or 1
            B[i] *= B[i - 1] or 1
        return max(max(nums), max(B))


def get_test_instance(example=1):
    nums = [2, 3, -2, 4]
    if example == 1:
        pass
    if example == 2:
        nums = [-2, 0, -1]
    return nums


def main():
    nums = get_test_instance(example=1)
    # nums = get_test_instance(example=1)
    res = Solution().maxProduct(nums)
    print(res)


if __name__ == '__main__':
    main()
