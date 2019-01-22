# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/23
"""  
Usage Of '64_min_path_sum.py' : 
"""


class Solution(object):
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        # 91.49%
        # 动态规划
        m = len(grid[0])
        n = len(grid)
        for i in range(1, m):
            grid[0][i] += grid[0][i - 1]
        for i in range(1, n):
            grid[i][0] += grid[i - 1][0]
        for i in range(1, n):
            for j in range(1, m):
                grid[i][j] = grid[i][j] + min(grid[i][j - 1], grid[i - 1][j])
        res = grid[n - 1][m - 1]
        return res


def get_test_instance(example=1):
    grid = [
        [1, 3, 1],
        [1, 5, 1],
        [4, 2, 1]
    ]
    if example == 1:
        pass
    if example == 2:
        grid = [[1]]
    return grid


def main():
    grid = get_test_instance(example=1)
    # grid = get_test_instance(example=2)
    res = Solution().minPathSum(grid)
    print(res)


if __name__ == '__main__':
    main()
