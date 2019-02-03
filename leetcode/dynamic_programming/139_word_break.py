# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/2/1
"""  
Usage Of '139_word_break.py' : 
"""


class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        # 40 ms - 55.64%
        # 记录每一个开始点
        if not s:
            return True

        breakp = [0]

        for i in range(len(s) + 1):
            for j in breakp:
                if s[j:i] in wordDict:
                    breakp.append(i)
                    break

        return breakp[-1] == len(s)

    def wordBreak2(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        # 28 ms - 97.67%
        l = set()
        for d in wordDict:
            l.add(len(d))

        dp = [0 for _ in s]
        for i in range(1, len(s) + 1):
            if s[0:i] in wordDict:
                dp[i - 1] = 1

        for i in range(1, len(s)):
            if dp[i] == 0:
                for k in l:
                    if i >= k and dp[i - k] == 1 and s[i - k + 1:i + 1] in wordDict:
                        dp[i] = 1
                        break

        return dp[-1] == 1

    def wordBreak3(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """

        # 递归超时

        def word_break(s1):
            if (s1 in wordDict) or (s1 == ''):
                return True
            else:
                for j in range(1, len(s1)):
                    if word_break(s1[:j]) and word_break(s1[j:]):
                        return True
                return False

        return word_break(s)


def get_test_instance(example=1):
    s = "leetcode"
    wordDict = ["leet", "code"]
    if example == 1:
        pass
    if example == 2:
        s = "applepenapple"
        wordDict = ["apple", "pen"]
    if example == 3:
        s = "catsandog"
        wordDict = ["cats", "dog", "sand", "and", "cat"]
    return s, wordDict


def main():
    s, wordDict = get_test_instance(example=1)
    # s, wordDict = get_test_instance(example=2)
    # s, wordDict = get_test_instance(example=3)
    res = Solution().wordBreak(s, wordDict)
    print(res)


if __name__ == '__main__':
    main()
