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
        pass


def get_test_instance(example=1):
    word1, word2 = 'horse', 'ros'  # 3
    if example == 1:
        pass
    if example == 2:
        word1 = "intention"
        word2 = "execution"  # 5
    return word1, word2


def main():
    word1, word2 = get_test_instance(example=1)
    # word1, word2 = get_test_instance(example=2)
    # word1, word2 = get_test_instance(example=3)
    res = Solution().minDistance(word1, word2)
    print(res)


if __name__ == '__main__':
    main()
