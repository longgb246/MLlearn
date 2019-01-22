# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/22
"""  
Usage Of '63_unique_paths_with_obstacles.py' : 
"""


class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        # 动态规划
        m = len(obstacleGrid[0])
        n = len(obstacleGrid)
        if (obstacleGrid[0][0] == 1) or (obstacleGrid[n - 1][m - 1] == 1):
            return 0
        dp = []
        flag = True
        # 初始化问题，防止初始路径绕过赋值
        for i in range(n):
            if (obstacleGrid[i][0] == 0) and flag:
                dp.append([1] + [0] * (m - 1))
            else:
                dp.append([0] * m)
                flag = False
        dp += [[0] * m]
        for i in range(n):
            for j in range(1, m):
                if obstacleGrid[i][j] == 0:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        res = dp[n - 1][m - 1]
        return res

    def uniquePathsWithObstacles2(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        # 99.71%
        # 尝试使用一维数组，依旧是动态规划
        m = len(obstacleGrid[0])
        n = len(obstacleGrid)
        if (obstacleGrid[0][0] == 1) or (obstacleGrid[n - 1][m - 1] == 1):
            return 0
        dp = [0] * m
        flag = True
        for i in range(n):
            if (obstacleGrid[i][0] == 0) and flag:
                dp[0] = 1
            else:
                dp[0] = 0
                flag = False
            for j in range(1, m):
                if obstacleGrid[i][j] == 0:
                    dp[j] += dp[j - 1]
                else:
                    dp[j] = 0
        res = dp[m - 1]
        return res


def get_test_instance(example=1):
    obstacleGrid = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
    if example == 1:
        pass
    if example == 2:
        obstacleGrid = [[1]]
    if example == 3:
        obstacleGrid = [[1, 0]]
    if example == 4:
        obstacleGrid = [[0], [1]]
    if example == 5:
        obstacleGrid = [[0, 0], [1, 0]]
    if example == 6:
        obstacleGrid = [[0, 0], [1, 1], [0, 0]]
    return obstacleGrid


def main():
    obstacleGrid = get_test_instance(example=1)
    # obstacleGrid = get_test_instance(example=2)
    # obstacleGrid = get_test_instance(example=3)
    # obstacleGrid = get_test_instance(example=4)
    # obstacleGrid = get_test_instance(example=5)
    # obstacleGrid = get_test_instance(example=6)
    res = Solution().uniquePathsWithObstacles2(obstacleGrid)
    print(res)


if __name__ == '__main__':
    main()
