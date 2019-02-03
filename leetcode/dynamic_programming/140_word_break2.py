# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/2/1
"""  
Usage Of '140_word_break2.py' : 
"""


class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """
        if not s:
            return True
        len_set = set(map(len, wordDict))

        dp = [[] for _ in range(len(s))]

        # for i in range(1, min(len(s), max(len_set))):
        #     if s[:i] in wordDict:
        #         dp[i - 1].append(s[:i])

        for i in range(len(s)):
            for k in len_set:
                pass
            pass


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
