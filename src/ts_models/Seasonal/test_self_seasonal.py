# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : lgb453476610@163.com
  Date    : 2018/10/11
  Usage   :
"""

import numpy as np
import pandas as pd
from dateutil.parser import parse


def get_max_ration_split(arr):
    """返回使得按分割点分割后，左右数据点集合平均值比值最大的分割点

    :param arr:
    :return:
    """
    rs = [np.mean(arr[0:i]) / np.mean(arr[i:]) for i in range(1, len(arr))]
    sp = np.argmax(rs) + 1
    return sp


def get_change_group(arr):
    """利用get_max_ration_split函数，生成 change point 分割后的分组

    :param arr:
    :return:
    arr:
    ------------------------------------------------
    1   20  30  15  4   5   23  45  20  1    2    4
    -------------------------------------------------
    """
    i = len(arr)
    # 生成 月值（对应月的销量）对
    month_value_t = zip(range(0, len(arr)), arr)
    """
    month_value_t:
    ------------------------------------------------
    1   2   3   4   5   6   7   8   9   10   11   12
    1   20  30  15  4   5   23  45  20  1    2    4
    -------------------------------------------------
    """
    # 按值（对应月的销量）排序
    sorted_values = sorted(month_value_t, key=lambda x: x[1], reverse=True)
    """
    sorted_values:
    ------------------------------------------------
    8   3   7   9   2   4   6   5   12   11   10   1
    45  30  23  20  20  15  5   4   4    2    1    1
    -------------------------------------------------
    """
    s_index, s_arr = zip(*sorted_values)
    r = []
    while i > 1:
        sub_arr = s_arr[0:i]
        index = get_max_ration_split(sub_arr)
        r.insert(0, zip(s_index[index:i], s_arr[index:i]))
        i = index
        if i <= 1:
            r.insert(0, zip(s_index[:i], s_arr[:i]))
    return r
    """
    r:
    -----------------------------------------------------
    8  | 3   7   9   2   4   6  | 5   12  | 11  | 10   1
    45 | 30  23  20  20  15  5  | 4   4   | 2   | 1    1
    -----------------------------------------------------
    """


def get_season_label(s, mh, gh, rh):
    """打标商品季节性（是否有季节性，某个月是否为旺季，最大销量的月）

    :param s: 某sku的每月销量list，假设s=[1,20,30,15,4,5,23,45,20,1,2,4]
    :param mh:旺季月份的最大个数，假设mh=6
    :param gh:seasonal peak的最大个数，假设gh=2
    :param rh:要求旺季月份的平均销量为总月平均销量的rh倍，假设rh=1
    :return:
    """
    # 最大销量月
    max_month = np.argmax(s) + 1
    ms = np.sum(s)
    cg = get_change_group(s)
    vg = []
    s_len = 0
    # 根据阈值mh（旺季月份的最大个数，设mh=6） 截取 分组
    """
    cg:
    -----------------------------------------------------
    8  | 3   7   9   2   4  | 6   5   12  | 11  | 10   1
    45 | 30  23  20  20  15 | 5   4   4   | 2   | 1    1
    -----------------------------------------------------
    """
    for i in range(len(cg)):
        g = cg[i]
        if s_len + len(g) <= mh:
            s_len += len(g)
            vg.append(g)
        else:
            break
    """
    vg:
    -----------------------
    8  | 3   7   9   2   4
    45 | 30  23  20  20  15
    -----------------------
    """
    # 合并有效销量
    m_list = []
    for g in vg:
        m_list.extend(g)
    """
    m_list :
    ----------------------
    8   3   7   9   2   4 
    45  30  23  20  20  15
    ----------------------
    """
    # 按月排序
    s_m_list = sorted(m_list, key=lambda x: x[0])
    """
    s_m_list :
    ----------------------
    2   3   4   7   8   9
    20  30  15  23  45  20
    ----------------------
    """
    # 合并连续月
    last_month = -2
    cmg = []
    for m in s_m_list:
        cm = m[0]
        if abs(cm - last_month) == 1:
            last_g.append(m)
        else:
            last_g = []
            last_g.append(m)
            cmg.append(last_g)
        last_month = cm
    # 循环合并 1月12(若第一个分组从1月开始，最后一个分组为12月，将最后一个分组与第一个分组合并)
    if cmg[0][0][0] == 0 and cmg[-1][0][0] == 11:
        cmg[0].extend(cmg[-1])
        cmg = cmg[0:-1]
    """
    cmg :
    ------------------------
    2   3   4  | 7   8   9
    20  30  15 | 23  45  20
    ------------------------
    """
    # 根据阈值gh(seasonal peak的最大个数，假设gh=2)截取分组
    valid_cmg = cmg[0:gh]
    """
    valid_cmg :
    ------------------------
    2   3   4  | 7   8   9
    20  30  15 | 23  45  20
    ------------------------
    """
    # 分区内的月tuple合并
    v_mv_list = []
    acc_mean = 0
    acc_m = 0
    for vg in valid_cmg:
        future_acc_month = len(vg) + acc_m
        future_acc_sum = acc_mean * acc_m + np.sum(vg, axis=0)[1]
        future_mean = future_acc_sum / float(future_acc_month)
        # 符合阈值，则加入分组
        """
        np.mean(s) * rh = 14.166666666666666 * 2
        """
        if future_mean > np.mean(s) * rh:
            # if future_mean > (ms - future_acc_sum) / (12 - future_acc_month) * rh:
            v_mv_list.extend(vg)
            acc_m = future_acc_month
            acc_mean = future_mean
    """
    v_mv_list:
    ----------
    7   8   9
    23  45  20
    ----------
    """
    # 初始化label
    label = [0] * 14
    label[13] = max_month
    if len(v_mv_list) > 0:
        label[0] = 1
    for vm in v_mv_list:
        label[vm[0] + 1] = 1
    """
    label = [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 8]
    """
    return label


def test_season():
    data_path = u'/Users/longguangbin/Work/Documents/SAAS/安踏线下/季节性/sample/duanku_sales.xls'
    data = pd.read_excel(data_path)
    data['commit_month'] = data['commit_date'].apply(lambda x: parse(x).strftime('%Y-%m'))
    data_month = data.groupby(['commit_month']).agg({'sku_num_sum': 'sum'}).reset_index()
    data_month['month'] = data_month['commit_month'].apply(lambda x: x.split('-')[1])
    data_month_all = data_month.groupby(['month']).agg({'sku_num_sum': 'sum'}).reset_index()
    get_season_label(data_month_all['sku_num_sum'].values, 6, 2, 1)
    pass
