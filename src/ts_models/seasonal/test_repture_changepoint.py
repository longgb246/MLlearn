# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : lgb453476610@163.com
  Date    : 2018/10/16
  Usage   :
"""

import sys
import platform

import pandas as pd

sys_platform = 'mac' if any(list(map(lambda x: x in platform.system().lower(), ('darwin', 'os2', 'os', 'mac')))) else \
    'win' if any(list(map(lambda x: x in platform.system().lower(), ('win32', 'cygwin', 'win')))) else 'other'

py_ver = sys.version_info[0]

if py_ver == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')
if py_ver == 3:
    long = int
    unicode = int

if sys_platform == 'win':
    from pylab import mpl

    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
elif sys_platform == 'mac':
    import matplotlib

    matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from dateutil.parser import parse

data_path = u'/Users/longguangbin/Work/Documents/SAAS/安踏线下/季节性/sample/duanku_sales.xls'
data = pd.read_excel(data_path)

data['commit_month'] = data['commit_date'].apply(lambda x: parse(x).strftime('%Y-%m'))
data_month = data.groupby(['commit_month']).agg({'sku_num_sum': 'sum'}).reset_index()

data['week_sale'] = data['sku_num_sum'].rolling(window=7).sum()
data.loc[:5, ['week_sale']] = data[:6]['sku_num_sum'].cumsum()
data['week_day'] = data['commit_date'].apply(lambda x: parse(x).weekday())
data_week = data[data['week_day'] == 6]

import ruptures as rpt


def test_1():
    # generate signal
    n_samples, dim, sigma = 1000, 3, 4
    n_bkps = 4  # number of breakpoints
    signal, bkps = rpt.pw_constant(n_samples, dim, n_bkps, noise_std=sigma)

    # detection
    # algo = rpt.Pelt(model="rbf").fit(signal[:,1])

    rpt_data = data_month['sku_num_sum'].values
    # rpt_data = data['sku_num_sum'].values

    algo = rpt.Pelt(model="rbf").fit(rpt_data)
    # algo = rpt.Binseg(model="rbf").fit(rpt_data)
    res_kps = algo.predict(pen=3)

    # display
    # rpt.display(signal, bkps, res_kps)
    rpt.display(rpt_data, res_kps)
    plt.show()


def test_win_base():
    signal = data_month['sku_num_sum'].values

    n, dim = 500, 3  # number of samples, dimension
    n_bkps, sigma = 3, 5  # number of change points, noise standart deviation
    signal, bkps = rpt.pw_constant(n, dim, n_bkps, noise_std=sigma)

    # change point detection
    model = "l2"  # "l1", "rbf", "linear", "normal", "ar"
    algo = rpt.Window(width=40, model=model).fit(signal)
    my_bkps = algo.predict(n_bkps=3)

    # show results
    rpt.show.display(signal, bkps, my_bkps, figsize=(10, 6))
    plt.show()

    # change point detection
    model = "l2"  # "l1", "rbf", "linear", "normal", "ar"
    algo = rpt.Window(width=40, model=model).fit(signal)
    my_bkps = algo.predict(n_bkps=3)

    # show results
    rpt.show.display(signal, my_bkps, figsize=(10, 6))
    plt.show()
    pass
