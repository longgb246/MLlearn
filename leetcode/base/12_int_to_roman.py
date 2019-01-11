# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/11
"""  
Usage Of '12_int_to_roman.py' : 
"""


class Solution:
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        m = [
            ['', 'M', 'MM', 'MMM'],
            ['', 'C', 'CC', 'CCC', 'CD', 'D', 'DC', 'DCC', 'DCCC', 'CM'],
            ['', 'X', 'XX', 'XXX', 'XL', 'L', 'LX', 'LXX', 'LXXX', 'XC'],
            ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX']
        ]

        d = [1000, 100, 10, 1]

        r = ''

        for i, v in enumerate(d):
            d, num = divmod(num, v)
            r += m[i][d]

        return r


def get_test_instance(example=1):
    num = 3
    if example == 1:
        pass
    if example == 2:
        num = 4
    if example == 3:
        num = 9
    if example == 4:
        num = 58
    if example == 5:
        num = 1994
    return num


def main():
    num = get_test_instance(example=1)
    # num = get_test_instance(example=2)
    # num = get_test_instance(example=3)
    # num = get_test_instance(example=4)
    # num = get_test_instance(example=5)
    res = Solution().intToRoman(num)
    print(res)


if __name__ == '__main__':
    main()
