# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/27
"""  
Usage Of '120_minimum_total.py' : 
"""


class Solution(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        # 32 ms - 84.64%
        # 动态规划
        if not triangle:
            return 0
        len_v = len(triangle)
        if len_v == 1:
            return triangle[0][0]
        dp = [0] * len_v
        dp[0] = triangle[0][0]
        for i in xrange(1, len_v):
            tmp_v = dp[0]
            dp[0] += triangle[i][0]
            for j in xrange(1, i):
                res = min(dp[j], tmp_v)
                tmp_v = dp[j]
                dp[j] = res + triangle[i][j]
            dp[i] = tmp_v + triangle[i][i]
        return min(dp)

    def minimumTotal2(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        # 更小内存方法：直接在原数组上进行，不用开辟新的空间 - 发现速度一样？
        L = len(triangle)
        if L == 1:
            return triangle[0][0]
        for i in range(1, L):
            for j in range(i + 1):
                left = 1e10 if j == 0 else triangle[i - 1][j - 1]
                mid = 1e10 if j == i else triangle[i - 1][j]
                triangle[i][j] += min(left, mid)
        return min(triangle[-1])


def get_test_instance(example=1):
    triangle = [
        [2],
        [3, 4],
        [6, 5, 7],
        [4, 1, 8, 3]
    ]
    if example == 1:
        pass
    if example == 2:
        S = "babgbag"
        T = "bag"
    return triangle


def main():
    triangle = get_test_instance(example=1)
    # triangle = get_test_instance(example=2)
    res = Solution().minimumTotal(triangle)
    print(res)


if __name__ == '__main__':
    main()
