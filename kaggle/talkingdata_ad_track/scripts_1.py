# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : lgb453476610@163.com
  Date    : 2018/11/6
  Usage   :
"""

from __future__ import print_function
import sys
import platform

# ------------------------ system platform ------------------------
# Get the run system platform
sys_platform = 'mac' if any(list(map(lambda x: x in platform.system().lower(), ('darwin', 'os2', 'os', 'mac')))) else \
    'win' if any(list(map(lambda x: x in platform.system().lower(), ('win32', 'cygwin', 'win')))) else 'other'
# Get the run python version
py_ver = sys.version_info[0]

# python2 reload sys to set utf8
if py_ver == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')
# python3 set long and unicode
if py_ver == 3:
    long = int
    unicode = int

# ------------------------ Matplotlib Setting ------------------------
if sys_platform == 'win':
    from pylab import mpl

    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
elif sys_platform == 'mac':
    import matplotlib

    matplotlib.use('TkAgg')

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.style.use('seaborn')

# ------------------------ Pandas Setting ------------------------
import pandas as pd

pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 180)  # 150
pd.set_option('display.max_columns', 40)

# ------------------------- Use -------------------------
import numpy as np
import pandas as pd
import datetime
import os
import time

import matplotlib.pyplot as plt
import seaborn as sns

# sns.set(rc={'figure.figsize': (12, 5)})

org_path = r'/Users/longguangbin/Work/scripts/kaggle/TalkingDataAdTracking'

for f in os.listdir(org_path):
    if 'zip' not in f:
        print(f.ljust(30) + str(round(os.path.getsize(org_path + os.sep + f) / 1000000, 2)) + 'MB')

df_train = pd.read_csv(org_path + os.sep + 'train.csv', nrows=1000000)
df_test = pd.read_csv(org_path + os.sep + 'test.csv', nrows=1000000)
