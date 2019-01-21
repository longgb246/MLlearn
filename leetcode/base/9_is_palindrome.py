# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/10
"""  
Usage Of 'is_palindrome.py' : 
"""


class Solution(object):
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        x_str = str(x)
        d, m = divmod(len(x_str), 2)
        res = True
        for i in range(d):
            if x_str[i] != x_str[-(i + 1)]:
                res = False
                break
        return res

    def isPalindrome2(self, x):
        res = str(x) == str(x)[::-1]
        return res


def get_test_instance(example=1):
    x = 121
    if example == 1:
        pass
    if example == 2:
        x = -121
    if example == 3:
        x = 10
    return x


def main():
    x = get_test_instance(example=1)
    # x = get_test_instance(example=2)
    # x = get_test_instance(example=3)
    res = Solution().isPalindrome(x)
    print(res)


if __name__ == '__main__':
    main()
