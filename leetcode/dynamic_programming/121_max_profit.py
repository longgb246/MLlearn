# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/27
"""  
Usage Of '121_max_profit.py' : 
"""


class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        # 32 ms	- 80.10% | 28 ms - 99.84%
        if not prices:
            return 0
        min_v = 1e10
        res = 0
        for i in prices:
            if min_v > i:
                min_v = i
            res = max(res, i - min_v)
        return res

    def maxProfit2(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        # 动态规划
        iCurrentMax = 0
        iFinalMax = 0
        for i in range(len(prices) - 1):
            iCurrentMax += prices[i + 1] - prices[i]
            if iCurrentMax < 0:
                iCurrentMax = 0
            if iCurrentMax > iFinalMax:
                iFinalMax = iCurrentMax
        return iFinalMax


def get_test_instance(example=1):
    prices = [7, 1, 5, 3, 6, 4]
    if example == 1:
        pass
    if example == 2:
        prices = [7, 6, 4, 3, 1]
    return prices


def main():
    prices = get_test_instance(example=1)
    # prices = get_test_instance(example=2)
    res = Solution().maxProfit(prices)
    print(res)


if __name__ == '__main__':
    main()
