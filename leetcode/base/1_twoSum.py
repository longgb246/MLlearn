# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2018/12/28
"""  
Usage Of 'sum_2' : 
"""


def two_sum(list_v, sum_v):
    """解多个答案的问题 """
    v_dict = {}
    success_pair = []
    success_v_pair = []

    # dict search
    for i, x in enumerate(list_v):
        if sum_v - x > 0:
            if (v_dict.get(x) is not None) and ([min(x, sum_v - x), max(x, sum_v - x)] not in success_pair):
                success_pair.append([min(x, sum_v - x), max(x, sum_v - x)])
            v_tmp = v_dict.get(sum_v - x, [])
            v_tmp.append(i)
            v_dict[sum_v - x] = v_tmp

    # generate all possible
    for min_v, max_v in success_pair:
        for left_v in v_dict[min_v]:
            for right_v in v_dict[max_v]:
                success_v_pair.append([min(left_v, right_v), max(left_v, right_v)])

    return success_v_pair


def two_sum_single(list_v, sum_v):
    """解单个答案的问题 """
    v_dict = {}
    success_pair = []

    # dict search
    for i, x in enumerate(list_v):
        if sum_v - x > 0:
            if v_dict.get(x) is not None:
                success_pair = [min(i, v_dict.get(x)), max(i, v_dict.get(x))]
                break
            else:
                v_dict[sum_v - x] = i

    return success_pair


def print_solution(list_v, sum_v, solution):
    print(solution)
    for l, r in solution:
        print('{0} = {1} + {2}'.format(sum_v, list_v[l], list_v[r]))


def main():
    sum_v = 68
    list_v = [2, 4, 65, 3, 4, 4, 62, 3, 35, 46, 234, 534, 5, 3245, 4, 345, 34, 25, 43, 52]
    solution = two_sum(list_v, sum_v)
    print_solution(list_v, sum_v, solution)


if __name__ == '__main__':
    main()
