# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/9
"""  
Usage Of '4_median_sorted_arrays.py' :
"""

import numpy as np


class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        return float(np.median(nums1 + nums2))


def get_test_instance(example=1):
    nums1 = [1, 3]
    nums2 = [2]
    if example == 1:
        # 2.0
        pass
    if example == 2:
        nums1 = [1, 2]
        nums2 = [3, 4]
        # 2.5
    return nums1, nums2


def main():
    nums1, nums2 = get_test_instance(example=1)
    # nums1, nums2 = get_test_instance(example=2)
    res = Solution().findMedianSortedArrays(nums1, nums2)
    print(res)


if __name__ == '__main__':
    main()
