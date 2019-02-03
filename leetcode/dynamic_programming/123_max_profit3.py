# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/27
"""  
Usage Of '123_max_profit3.py' :
"""


class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        import sys
        first_buy, first_sell, second_buy, second_sell = -sys.maxsize, 0, -sys.maxsize, 0
        for price in prices:
            first_buy = max(first_buy, -price)  # 第一次买入手上的钱
            first_sell = max(first_sell, price + first_buy)  # 第一次卖出手上的钱
            second_buy = max(second_buy, first_sell - price)  # 第二次买入手上的钱
            second_sell = max(second_sell, price + second_buy)  # 第二次卖出手上的钱
        return second_sell


def get_test_instance(example=1):
    prices = [3, 3, 5, 0, 0, 3, 1, 4]
    if example == 1:
        pass
    if example == 2:
        prices = [1, 2, 3, 4, 5]
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
