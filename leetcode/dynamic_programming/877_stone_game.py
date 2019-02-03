# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/27
"""  
Usage Of '877_stone_game.py' : 
"""


class Solution(object):
    def stoneGame(self, piles):
        """
        :type piles: List[int]
        :rtype: bool
        """
        # 24 ms - 99.06%
        # 显然，亚历克斯总是赢得 2 堆时的游戏。 通过一些努力，我们可以获知她总是赢得 4 堆时的游戏。
        # 如果亚历克斯最初获得第一堆，她总是可以拿第三堆。 如果她最初取到第四堆，她总是可以取第二堆。第一+第三，第二+第四 中的至少一组是更大的，所以她总能获胜。
        # 我们可以将这个想法扩展到 N 堆的情况下。设第一、第三、第五、第七桩是白色的，第二、第四、第六、第八桩是黑色的。 亚历克斯总是可以拿到所有白色桩或所有黑色桩，其中一种颜色具有的石头数量必定大于另一种颜色的。
        return True


def get_test_instance(example=1):
    piles = [5, 3, 4, 5]
    if example == 1:
        pass
    if example == 2:
        prices = [1, 2, 3, 4, 5]
    if example == 2:
        prices = [7, 6, 4, 3, 1]
    return piles


def main():
    prices = get_test_instance(example=1)
    # prices = get_test_instance(example=2)
    res = Solution().stoneGame(prices)
    print(res)


if __name__ == '__main__':
    main()
