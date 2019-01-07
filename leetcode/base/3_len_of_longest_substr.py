# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/7
"""  
Usage Of '3_len_of_longest_substr' : 
"""


class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        # s = 'abcabcbb'
        left = 0
        res = 0
        map_r = {}
        for i in range(len(s)):
            if map_r.has_key(s[i]):
                left = max(left, map_r[s[i]] + 1)
            res = max(res, i - left + 1)
            map_r[s[i]] = i
        return res


def get_test_instance(example=1):
    str_1 = 'abcabcbb'
    if example == 1:
        str_1 = 'abcabcbb'
    if example == 2:
        str_1 = 'bbbbb'
    if example == 3:
        str_1 = 'pwwkew'
    return str_1


def main():
    s1 = get_test_instance(example=1)
    # s1 = get_test_instance(example=2)
    # s1 = get_test_instance(example=3)
    res = Solution().lengthOfLongestSubstring(s1)
    print(res)


if __name__ == '__main__':
    main()
