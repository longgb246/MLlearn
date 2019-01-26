# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/26
"""  
Usage Of '87_is_scramble.py' : 
"""

from collections import Counter


class Solution(object):
    def isScramble(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        # 104 ms - 22.58%
        # 递归
        # 频率统计直方图剪枝
        if Counter(s1) != Counter(s2):
            return False
        if s1 == s2:
            return True
        for i in range(1, len(s1)):
            if self.isScramble(s1[i:], s2[i:]) and self.isScramble(s1[:i], s2[:i]):
                return True
            if self.isScramble(s1[i:], s2[:-i]) and self.isScramble(s1[:i], s2[-i:]):
                return True
        return False

    def isScramble2(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        # 40 ms - 100.00%
        # 递归，加一个字典记录重复子串

        mem = {}

        def solve(w1, w2):
            # mem 用于记录，当有重复的子串的时候，跳过判断
            if (w1, w2) not in mem:
                if w1 == w2:
                    return True
                if sorted(w1) != sorted(w2):
                    return False
                else:
                    l = len(w1)
                    for k in xrange(1, l):
                        if solve(w1[:k], w2[:k]) and solve(w1[k:], w2[k:]):
                            mem[(w1, w2)] = True
                            break
                        if solve(w1[k:], w2[:l - k]) and solve(w1[:k], w2[l - k:]):
                            mem[(w1, w2)] = True
                            break
                    else:  # 当不触发 break 的时候运行，即没有找到解
                        mem[(w1, w2)] = False
            return mem[(w1, w2)]

        return solve(s1, s2)


def get_test_instance(example=1):
    s1 = "great"
    s2 = "rgeat"
    if example == 1:
        pass
    if example == 2:
        s1 = "abcde"
        s2 = "caebd"
    return s1, s2


def main():
    # s1, s2 = get_test_instance(example=1)
    s1, s2 = get_test_instance(example=2)
    res = Solution().isScramble2(s1, s2)
    print(res)


if __name__ == '__main__':
    main()
