# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/23
"""  
Usage Of '72_min_distance.py' : 
"""


class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        # 26.40%
        # 动态规划 - 递推公式
        n, m = len(word1) + 1, len(word2) + 1
        dp = [[0 for _ in range(m)] for _ in range(n)]

        for i in range(n):
            dp[i][0] = i
        for i in range(m):
            dp[0][i] = i

        for i in range(1, n):
            for j in range(1, m):
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1)
                dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] + (0 if word1[i - 1] == word2[j - 1] else 1))

        return dp[n - 1][m - 1]

    def minDistance2(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        # 97.46%
        # 使用一维数组
        max_word = word1 if len(word1) > len(word2) else word2
        min_word = word1 if len(word1) <= len(word2) else word2

        len_max = len(max_word)
        len_min = len(min_word)

        if len_min == 0:
            return len_max

        dp = range(len_min + 1)

        for i in range(1, len_max + 1):
            dp[0] = i
            tmp_v = i - 1
            for j in range(1, len_min + 1):
                tmp_t = min(dp[j], dp[j - 1], tmp_v + (-1 if max_word[i - 1] == min_word[j - 1] else 0))
                tmp_v = dp[j]
                dp[j] = tmp_t + 1

        return dp[len_min]


def get_test_instance(example=1):
    word1, word2 = 'horse', 'ros'  # 3
    if example == 1:
        pass
    if example == 2:
        word1 = "intention"
        word2 = "execution"  # 5
    if example == 3:
        word1 = "a"
        word2 = "ab"
    return word1, word2


def main():
    word1, word2 = get_test_instance(example=1)
    # word1, word2 = get_test_instance(example=2)
    # word1, word2 = get_test_instance(example=3)
    res = Solution().minDistance2(word1, word2)
    print(res)


if __name__ == '__main__':
    main()
