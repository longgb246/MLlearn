# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : lgb453476610@163.com
  Date    : 2018/10/5
  Usage   :
"""

from __future__ import print_function
import datetime
import numpy as np
import pandas as pd
import statsmodels as sm
import matplotlib
from dateutil.parser import parse

matplotlib.use('TkAgg')  # 使用tk画图
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def date_calculate(start_date, cal_date=0):
    """From the start date to a certain direction to get a date list.

    :param start_date: Start date to calculate
    :param cal_date: From the start date to a certain direction.
    :return: list

    Example
    ----------
    >>> MyTools.date_calculate('2017-03-04', 3)
    ['2017-03-04', '2017-03-05', '2017-03-06', '2017-03-07']
    >>> MyTools.date_calculate('2017-03-04', 0)
    ['2017-03-04']
    >>> MyTools.date_calculate('2017-03-04', -3)
    ['2017-03-01', '2017-03-02', '2017-03-03', '2017-03-04']
    """
    start_date_dt = parse(start_date)
    end_date_dt = start_date_dt + datetime.timedelta(cal_date)
    min_date = min(start_date_dt, end_date_dt)
    max_date = max(start_date_dt, end_date_dt)
    date_range = map(lambda x: (min_date + datetime.timedelta(x)).strftime('%Y-%m-%d'),
                     range((max_date - min_date).days + 1))
    return date_range


def get_sample_data():
    sample_data = [10930, 10318, 10595, 10972, 7706, 6756, 9092, 10551, 9722, 10913, 11151, 8186, 6422,
                   6337, 11649, 11652, 10310, 12043, 7937, 6476, 9662, 9570, 9981, 9331, 9449, 6773, 6304, 9355,
                   10477, 10148, 10395, 11261, 8713, 7299, 10424, 10795, 11069, 11602, 11427, 9095, 7707, 10767,
                   12136, 12812, 12006, 12528, 10329, 7818, 11719, 11683, 12603, 11495, 13670, 11337, 10232,
                   13261, 13230, 15535, 16837, 19598, 14823, 11622, 19391, 18177, 19994, 14723, 15694, 13248,
                   9543, 12872, 13101, 15053, 12619, 13749, 10228, 9725, 14729, 12518, 14564, 15085, 14722,
                   11999, 9390, 13481, 14795, 15845, 15271, 14686, 11054, 10395]
    date_list = date_calculate('2001', cal_date=89)
    data = pd.DataFrame(zip(date_list, sample_data), columns=['dt', 'value'])
    return data


def plot_series(x, y):
    plt.style.use('seaborn-darkgrid')
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    x_tick_labels = list(ax.get_xticklabels())
    tick_num = 10  # 刻度数目
    # tick_spacing = int(np.ceil(len(x_tick_labels) * 1.0 / tick_num))
    print(len(x_tick_labels) * 1.0)
    print(len(x_tick_labels) * 1.0 / tick_num)
    tick_spacing = len(x_tick_labels) / tick_num
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 0.55))
    plt.show()


def tmp():
    data = get_sample_data()
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    ax.plot(data['dt'], data['value'])

    plot_series(data['dt'], data['value'])
    pass


if __name__ == '__main__':
    pass
