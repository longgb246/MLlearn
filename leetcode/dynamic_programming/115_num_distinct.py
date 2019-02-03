# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/27
"""  
Usage Of '115_num_distinct.py' : 
"""


class Solution(object):
    def numDistinct(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: int
        """
        # 120 ms - 67.09%
        # 动态规划
        len_s = len(s)
        len_t = len(t)
        if len_s < len_t:
            return 0
        if (len_t == 0) or (len_s == 0 and len_t == 0):
            return 1
        dp = [1] * (len_s + 1)
        for i in xrange(len_t):
            tmp_v = dp[i]
            dp[:(i + 1)] = [0] * (i + 1)
            for j in xrange(i + 1, len_s + 1):
                if t[i] == s[j - 1]:
                    res = dp[j - 1] + tmp_v
                    tmp_v = dp[j]
                    dp[j] = res
                else:
                    tmp_v = dp[j]
                    dp[j] = dp[j - 1]
        return dp[-1]

    def numDistinct2(self, S, T):
        # 28 ms - 100.00%
        # 记忆化回溯法可以低效率的通过
        # 尝试转化为增子序列来解决问题
        from collections import Counter
        # 统计T中每个字母的出现次数
        ct = Counter(T)
        # 序列字典
        dt = {it[0]: [] for it in ct.items()}
        # 将S中每一个T中出现了的字母的index存入数组
        for i, w in enumerate(S):
            if w in dt:
                dt[w] += [i]
        for w in ct:
            if len(dt[w]) < ct[w]:
                return 0
        # 依据T的顺序存储的索引数组
        nt = [dt[w][:] for w in T]
        # 卡最小索引
        low_limit = -1
        for i in range(len(T)):
            while nt[i][0] <= low_limit:
                nt[i] = nt[i][1:]
                if not nt[i]:
                    return 0
            low_limit = nt[i][0]
        # 卡最大索引
        up_limit = len(S)
        for i in range(len(T))[::-1]:
            while nt[i][-1] >= up_limit:
                nt[i] = nt[i][:-1]
                if not nt[i]:
                    return 0
            up_limit = nt[i][-1]
        # 最终统计, 最后一个用于收集结果
        nt += [[len(S)]]
        # 默认选用最小的index
        # 第一个字母的所有选值视为一种情形
        results = [1 for _ in nt[0]]
        # 然后按T的字母顺序向下推进，将所有的情形加起来
        while len(nt) > 1:
            count, new_results = 0, []
            pre, nxt = nt[0], nt[1]
            # print results
            # print "->", results, pre, nxt
            for i, x in enumerate(results):
                while pre[0] >= nxt[0]:
                    new_results += [count]
                    nxt = nxt[1:]
                pre = pre[1:]
                count += x
            for _ in nxt:
                new_results += [count]
            results = new_results
            # print "<-", results, nxt
            nt = nt[1:]
        return results[0]


def get_test_instance(example=1):
    S = "rabbbit"
    T = "rabbit"
    if example == 1:
        pass
    if example == 2:
        S = "babgbag"
        T = "bag"
    return S, T


def main():
    S, T = get_test_instance(example=1)
    # S, T = get_test_instance(example=2)
    res = Solution().numDistinct(S, T)
    print(res)


if __name__ == '__main__':
    main()
