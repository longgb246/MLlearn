# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/26
"""  
Usage Of '97_is_interleave.py' : 
"""


class Solution(object):
    def isInterleave(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        """
        # 36 ms - 70.27%
        # 动态规划
        len1 = len(s1)
        len2 = len(s2)
        len3 = len(s3)

        if len1 + len2 != len3:
            return False
        dp = [[False] * (len2 + 1) for _ in range(len1 + 1)]
        dp[0][0] = True
        for i in xrange(len1 + 1):
            for j in xrange(len2 + 1):
                if j > 0:  # 分别能更新到边界的值
                    dp[i][j] = dp[i][j - 1] and (s3[i + j - 1] == s2[j - 1])
                if i > 0:
                    dp[i][j] = dp[i][j] or (dp[i - 1][j] and s3[i + j - 1] == s1[i - 1])
        return dp[len1][len2]

    def isInterleave2(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        """
        # 28 ms - 97.30%
        # 递归
        if len(s1) + len(s2) != len(s3):
            return False

        rbuf = {}

        def test(l1, l2):
            if l1 == 0 and l2 == 0:
                return True
            if l1 < 0 or l2 < 0:
                return False
            if (l1, l2) in rbuf:
                return rbuf[(l1, l2)]
            if l1 == 0:
                ret = (s2[l2 - 1] == s3[l2 - 1]) and test(l1, l2 - 1)
            elif l2 == 0:
                ret = (s1[l1 - 1] == s3[l1 - 1]) and test(l1 - 1, l2)
            else:
                ret = ((s1[l1 - 1] == s3[l1 + l2 - 1]) and test(l1 - 1, l2)) or \
                      ((s2[l2 - 1] == s3[l1 + l2 - 1]) and test(l1, l2 - 1))
            rbuf[(l1, l2)] = ret
            return ret

        return test(len(s1), len(s2))


def get_test_instance(example=1):
    s1 = "aabcc"
    s2 = "dbbca"
    s3 = "aadbbcbcac"
    if example == 1:
        pass
    if example == 2:
        s1 = "aabcc"
        s2 = "dbbca"
        s3 = "aadbbbaccc"
    return s1, s2, s3


def main():
    s1, s2, s3 = get_test_instance(example=1)
    # s1, s2, s3 = get_test_instance(example=2)
    res = Solution().isInterleave2(s1, s2, s3)
    print(res)


if __name__ == '__main__':
    main()
