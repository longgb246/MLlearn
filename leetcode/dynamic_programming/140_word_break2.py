# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/2/1
"""  
Usage Of '140_word_break2.py' : 
"""


class Solution(object):
    def can_word_break(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
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

        return dp

    def DFS(self, s, wordDict, res, tmp):
        if not s:
            res.append(tmp.strip())
        for word in wordDict:
            if s.startswith(word):
                self.DFS(s[len(word):], wordDict, res, tmp + word + " ")
            else:
                continue

    def wordBreak(self, s, wordDict):
        # 44 ms - 89.02%
        if not s:
            return []
        res = self.can_word_break(s, wordDict)

        new_res = []
        if bool(res[-1]):
            self.DFS(s, wordDict, new_res, "")
        return new_res

    def wordBreak2(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """

        # 28 ms - 100.00%

        def sentence(s, wordDict, wordLen, i, memo):
            if i in memo:
                return memo[i]
            memo[i] = [s[i:i + l] + (tail and ' ' + tail)
                       for l in wordLen
                       if s[i:i + l] in wordDict
                       for tail in sentence(s, wordDict, wordLen, i + l, memo)]
            return memo[i]

        wordDict = set(wordDict)
        wordLen = set([len(word) for word in wordDict])
        memo = {}
        memo[len(s)] = ['']
        return sentence(s, wordDict, wordLen, 0, memo)

    def wordBreak3(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """
        if not s and not self.can_word_break(s, wordDict)[-1] == 1:
            return []

        dp = [[] for _ in range(len(s) + 1)]
        dp[0] = ['']

        for i in range(len(s)):
            for w in wordDict:
                len_w = len(w)
                if ((i + 1) >= len_w) and (dp[i - len_w + 1]) and (s[(i - len_w + 1):(i + 1)] == w):
                    dp[i + 1].extend([v + ' ' + w for v in dp[i - len_w + 1]])

        return map(lambda x: x.strip(), dp[-1])


def get_test_instance(example=1):
    s = "catsanddog"
    wordDict = ["cat", "cats", "and", "sand", "dog"]
    if example == 1:
        pass
    if example == 2:
        s = "pineapplepenapple"
        wordDict = ["apple", "pen", "applepen", "pine", "pineapple"]
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
