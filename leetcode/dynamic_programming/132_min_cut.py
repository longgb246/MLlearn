# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/27
"""  
Usage Of '132_min_cut.py' : 
"""


class Solution(object):
    def minCut(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 164 ms - 86.59%
        # 动态规划
        # 因为 case 里面很多都是0、1的，所以加上下面这句话，会提升极多。（ 不加 - 1348 ms - 8.54% ）
        if s == s[::-1]:
            return 0
        for i in range(1, len(s)):
            if s[:i] == s[:i][::-1] and s[i:] == s[i:][::-1]:
                return 1

        s_len = len(s)
        mem = [i for i in range(-1, s_len)]

        for i in range(1, s_len + 1):
            for j in range(i):
                if s[j:i] == s[j:i][::-1]:
                    mem[i] = min(mem[i], mem[j] + 1)

        return mem[-1]

    def minCut2(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 28 ms - 100.00%
        if s == s[::-1]:
            return 0
        for i in range(1, len(s)):
            if s[:i] == s[:i][::-1] and s[i:] == s[i:][::-1]:
                return 1
        # algorithm
        cut = [x for x in range(-1, len(s))]  # cut numbers in worst case (no palindrome)
        for i in range(len(s)):
            r1, r2 = 0, 0
            # use i as origin, and gradually enlarge radius if a palindrome exists
            # odd palindrome
            while i - r1 >= 0 and i + r1 < len(s) and s[i - r1] == s[i + r1]:
                cut[i + r1 + 1] = min(cut[i + r1 + 1], cut[i - r1] + 1)
                r1 += 1
            # even palindrome
            while i - r2 >= 0 and i + r2 + 1 < len(s) and s[i - r2] == s[i + r2 + 1]:
                cut[i + r2 + 2] = min(cut[i + r2 + 2], cut[i - r2] + 1)
                r2 += 1
        return cut[-1]


def get_test_instance(example=1):
    s = "aab"
    if example == 1:
        pass
    if example == 2:
        s = [1, 2, 3, 4, 5]
    return s


def main():
    s = get_test_instance(example=1)
    # s = get_test_instance(example=2)
    res = Solution().minCut(s)
    print(res)


if __name__ == '__main__':
    main()
