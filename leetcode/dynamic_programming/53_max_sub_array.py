# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/21
"""  
Usage Of '53_max_sub_array.py' : 
"""


class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        p = [0] * len(nums)
        p[0] = nums[0]
        i = 1
        for num in nums[1:]:
            p[i] = max(0, p[i - 1]) + num
            i += 1
        res = max(p)
        return res


def get_test_instance(example=1):
    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    if example == 1:
        pass
    if example == 2:
        nums = [-2, -1]
    return nums


def main():
    nums = get_test_instance(example=1)
    # nums = get_test_instance(example=2)
    res = Solution().maxSubArray(nums)
    print(res)


if __name__ == '__main__':
    main()
