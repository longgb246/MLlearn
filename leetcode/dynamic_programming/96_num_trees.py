# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/26
"""  
Usage Of '96_num_trees.py' : 
"""


class Solution(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 24 ms - 96.06%
        if n == 0:
            return 0
        if n == 1:
            return 1
        if n == 2:
            return 2
        dp = [1, 1, 2]
        for i in xrange(3, n + 1):
            dp.append(0)
            for j in xrange(i):
                dp[i] += (dp[j] * dp[i - j - 1])
        return dp[-1]


def get_test_instance(example=1):
    n = 3
    if example == 1:
        pass
    if example == 2:
        n = "226"
    return n


def main():
    n = get_test_instance(example=1)
    # n = get_test_instance(example=2)
    res = Solution().numTrees(n)
    print(res)


if __name__ == '__main__':
    main()
