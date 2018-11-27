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
import lgblearn
from lgblearn import plot as lpt
import os

org_path = r'/Users/longguangbin/Work/scripts/kaggle/TalkingDataAdTracking'
files_list = map(lambda x: org_path + os.sep + x, os.listdir(org_path))

# 检查文件
lgblearn.eda.summary_files(files_list)
# 文件结果
# SIZE       LINES       FILE
# 6.0KB      0           /Users/longguangbin/Work/scripts/kaggle/TalkingDataAdTracking/.DS_Store
# 823.28MB   18790470    /Users/longguangbin/Work/scripts/kaggle/TalkingDataAdTracking/test.csv
# 3.89MB     100001      /Users/longguangbin/Work/scripts/kaggle/TalkingDataAdTracking/train_sample.csv
# 2.48GB     57537506    /Users/longguangbin/Work/scripts/kaggle/TalkingDataAdTracking/test_supplement.csv
# 7.02GB     184903891   /Users/longguangbin/Work/scripts/kaggle/TalkingDataAdTracking/train.csv
# 186.52MB   18790470    /Users/longguangbin/Work/scripts/kaggle/TalkingDataAdTracking/sample_submission.csv

df_train = pd.read_csv(org_path + os.sep + 'train.csv', nrows=1000000)
df_test = pd.read_csv(org_path + os.sep + 'test.csv', nrows=1000000)

cols = ['ip', 'app', 'device', 'os', 'channel']
uniques = [len(df_train[col].unique()) for col in cols]

import seaborn as sns

plt.figure(figsize=(15, 8))
sns.set(font_scale=1.2)
# ax = sns.barplot(cols, uniques, palette=pal, log=True)
ax = sns.barplot(cols, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 10, uniq, ha="center")

lpt.barplot(cols, uniques, log=True)
lpt.barplot(cols, uniques)
