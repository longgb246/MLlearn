# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/9
"""  
Usage Of '6_convert.py' : 
"""

from string import Template


def transpose(matrix):
    return zip(*matrix)


class Solution(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        len_s = len(s)
        if numRows == 1:
            return s
        elif numRows == 2:
            d, m = divmod(len_s, 2)
            res = ''.join([s[x * 2] for x in range(d + m)] + [s[x * 2 + 1] for x in range(d)])
            return res
        else:
            median_v = numRows - 2
            split_num = median_v * 2 + 2
            len_t = Template('{:${len}}').substitute(len=numRows)
            len_t_r = Template('{:>${len}}').substitute(len=numRows)
            d, m = divmod(len_s, split_num)
            m_v = 1 if m > 0 else 0
            res = []
            for x in range(d + m_v):
                all_s = s[(x * split_num):((x + 1) * split_num)]
                left = len_t.format(all_s[:numRows])
                right = len_t_r.format(all_s[numRows:][::-1] + ' ')
                res.append(map(lambda x: x[0] + x[1], zip(left, right)))
            res_t = transpose(res)
            res_v = ''.join(map(lambda x: ''.join(x), res_t)).replace(' ', '')
            return res_v


def get_test_instance(example=1):
    s = "LEETCODEISHIRING"
    numRows = 3
    if example == 1:
        # "LCIRETOESIIGEDHN"
        pass
    if example == 2:
        s = "LEETCODEISHIRING"
        numRows = 4
        # "LDREOEIIECIHNTSG"
    return s, numRows


def main():
    s, numRows = get_test_instance(example=1)
    # s, numRows = get_test_instance(example=2)
    res = Solution().convert(s, numRows)
    print(res)


if __name__ == '__main__':
    main()
