# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : lgb453476610@163.com
  Date    : 2018/10/16
  Usage   :
"""

import sys
from dateutil.parser import parse
import platform

import numpy as np
import pandas as pd

sys_platform = 'mac' if platform.system().lower() in ('darwin', 'os2', 'os2emx') else \
    'win' if platform.system().lower() in ('win32', 'cygwin') else 'other'

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
    from matplotlib.font_manager import FontProperties

    matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.style.use('seaborn')

data_path = u'/Users/longguangbin/Work/Documents/SAAS/安踏线下/季节性/sample/duanku_sales.xls'
data = pd.read_excel(data_path)

import ruptures as rpt

# # generate signal
n_samples, dim, sigma = 1000, 3, 4
n_bkps = 4  # number of breakpoints
signal, bkps = rpt.pw_constant(n_samples, dim, n_bkps, noise_std=sigma)

# detection
# algo = rpt.Pelt(model="rbf").fit(signal[:,1])
algo = rpt.Pelt(model="rbf").fit(data['sku_num_sum'].values)
result = algo.predict(pen=10)

# display
rpt.display(signal, bkps, result)
rpt.display(data['sku_num_sum'].values, result)
plt.show()
