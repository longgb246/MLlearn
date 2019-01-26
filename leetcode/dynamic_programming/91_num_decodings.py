# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/26
"""  
Usage Of '91_num_decodings.py' : 
"""


class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 36 ms - 45.60%
        # 动态规划
        if (not s) or (int(s[0]) == 0):
            return 0
        if len(s) == 1:
            return 1
        else:  # 初始化前 2 个状态
            v_1 = int(s[1])
            v_2 = int(s[:2])
            if v_1 == 0:
                if v_2 > 20:
                    return 0
                else:
                    dp = [1, 1]
            elif v_2 <= 26:
                dp = [1, 2]
            else:
                dp = [1, 1]
        # 新增状态仅与前2个状态有关
        for i in xrange(2, len(s)):
            v_2 = int(s[(i - 1):(i + 1)])
            cond1 = dp[i - 1] if int(s[i]) > 0 else 0  # 当前值比0大
            cond2 = dp[i - 2] if (v_2 <= 26) and (v_2 >= 10) else 0  # 加上前一个值小于26
            if any([cond1, cond2]):
                dp.append(cond1 + cond2)
            else:
                return 0
        return dp[-1]

    def numDecodings2(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 28 ms - 99.45%
        if s[0] == '0':
            return 0
        l = len(s)
        dp = [1]
        for i in range(1, l):
            cond1 = dp[i - 1] if s[i] != '0' else 0
            cond2 = (dp[i - 2] if i > 1 else 1) if '09' < s[i - 1:i + 1] < '27' else 0
            if any([cond1, cond2]):
                dp.append(cond1 + cond2)
            else:
                return 0
        return dp[l - 1]


def get_test_instance(example=1):
    s = "12"
    if example == 1:
        pass
    if example == 2:
        s = "226"
    if example == 3:
        s = "0"
    if example == 4:
        s = "101"
    if example == 5:
        s = "110"
    if example == 6:
        s = "301"
    if example == 7:
        s = "27"
    if example == 8:
        s = "20"
    return s


def main():
    s = get_test_instance(example=1)
    # s = get_test_instance(example=2)
    # s = get_test_instance(example=3)
    # s = get_test_instance(example=4)
    # s = get_test_instance(example=5)
    # s = get_test_instance(example=6)
    # s = get_test_instance(example=7)
    # s = get_test_instance(example=8)
    res = Solution().numDecodings2(s)
    print(res)


if __name__ == '__main__':
    main()
