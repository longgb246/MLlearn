# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/22
"""  
Usage Of '62_unique_paths.py' : 
"""


class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        # 是一个组合问题，必然会走 m+n-2 步，原问题等价为排列组合的 C(m+n-2)(n-1) 的问题。
        # 即为在 m+n-2 步中，选出 n-1 的向下走的位置，计算方便为选择 m,n 中较小的那个。
        min_v = min(m, n)
        upper = 1
        lower = 1
        for i in range(min_v - 1):
            upper *= (m + n - 2 - i)
            lower *= (min_v - 1 - i)
        res = int(upper / lower)
        return res

    def uniquePaths2(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        # 动态规划: (原) dp[i][j] = dp[i-1][j] + dp[i][j-1] 步数的递推公式。
        # 可以改为意味数组进行遍历: dp[j] += dp[j-1]，j-1 保留了上次的信息。
        min_v = min(m, n)
        max_v = max(m, n)
        dp = [0] * min_v
        dp[0] = 1
        for i in range(max_v):
            for j in range(1, min_v):
                dp[j] += dp[j - 1]
        res = dp[min_v - 1]
        return res


def get_test_instance(example=1):
    m = 3
    n = 2
    if example == 1:
        pass
    if example == 2:
        m = 7
        n = 3
    return m, n


def main():
    m, n = get_test_instance(example=1)
    # m, n = get_test_instance(example=2)
    res = Solution().uniquePaths(m, n)
    print(res)


if __name__ == '__main__':
    main()
