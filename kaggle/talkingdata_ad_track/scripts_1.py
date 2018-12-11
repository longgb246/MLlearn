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
import gc
import numpy as np

org_path = r'/Users/longguangbin/Work/scripts/kaggle/TalkingDataAdTracking'


# 浅紫色：#a675a1
# 浅绿色：#75a1a6


def eda_test_1():
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

    # import seaborn as sns
    #
    # plt.figure(figsize=(15, 8))
    # sns.set(font_scale=1.2)
    # # ax = sns.barplot(cols, uniques, palette=pal, log=True)
    # ax = sns.barplot(cols, uniques, log=True)
    # ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')
    # for p, uniq in zip(ax.patches, uniques):
    #     height = p.get_height()
    #     ax.text(p.get_x() + p.get_width() / 2., height + 10, uniq, ha="center")

    lpt.barplot(cols, uniques, log=True)
    lpt.barplot(cols, uniques)

    # ip, app, device, os and channel are actually categorical variables encoded as integers. Set them as categories for analysis.
    variables = ['ip', 'app', 'device', 'os', 'channel']
    for v in variables:
        df_train[v] = df_train[v].astype('category')
        df_test[v] = df_test[v].astype('category')

    train = df_train
    test = df_test

    # Convert date stamps to date/time type.
    # set click_time and attributed_time as timeseries
    train['click_time'] = pd.to_datetime(train['click_time'])
    train['attributed_time'] = pd.to_datetime(train['attributed_time'])
    test['click_time'] = pd.to_datetime(test['click_time'])

    # set as_attributed in train as a categorical
    train['is_attributed'] = train['is_attributed'].astype('category')

    # Now lets do a quick inspection of train and test data main statistics
    train.describe()

    # double check that 'attributed_time' is not Null for all values that resulted in download (i.e. is_attributed == 1)
    train[['attributed_time', 'is_attributed']][train['is_attributed'] == 1].describe()

    # set click_id to categorical, for cleaner statistics view
    test['click_id'] = test['click_id'].astype('category')
    test.describe()

    # temporary table to see ips with their associated count frequencies
    temp = train['ip'].value_counts().reset_index(name='counts')
    temp.columns = ['ip', 'counts']
    temp[:10]

    # add temporary counts of ip feature ('counts') to the train table, to see if IPs with high counts have conversions
    train = train.merge(temp, on='ip', how='left')
    # check top 10 values
    train[train['is_attributed'] == 1].sort_values('counts', ascending=False)[:10]

    # convert 'is_attributed' back to numeric for proportion calculations
    train['is_attributed'] = train['is_attributed'].astype(int)

    # 检查 y 与一些 x 之间的关系。是否有直接的关系
    # Conversion rates over Counts of 300 most popular IPs

    train_smp = pd.read_csv(org_path + os.sep + 'train.csv', nrows=1000000)
    # convert click_time and attributed_time to time series
    train_smp['click_time'] = pd.to_datetime(train_smp['click_time'])
    train_smp['attributed_time'] = pd.to_datetime(train_smp['attributed_time'])

    # round the time to nearest hour
    train_smp['click_rnd'] = train_smp['click_time'].dt.round('H')

    # check for hourly patterns
    train_smp[['click_rnd', 'is_attributed']].groupby(['click_rnd'], as_index=True).count().plot()
    plt.title('HOURLY CLICK FREQUENCY')
    plt.ylabel('Number of Clicks')

    train_smp[['click_rnd', 'is_attributed']].groupby(['click_rnd'], as_index=True).mean().plot()
    plt.title('HOURLY CONVERSION RATIO')
    plt.ylabel('Converted Ratio')

    # extract hour as a feature
    train_smp['click_hour'] = train_smp['click_time'].dt.hour

    train_smp[['click_hour', 'is_attributed']].groupby(['click_hour'], as_index=True).count(). \
        plot(kind='bar', color='#a675a1')
    plt.title('HOURLY CLICK FREQUENCY Barplot')
    plt.ylabel('Number of Clicks')

    # adapted from https://stackoverflow.com/questions/9103166/multiple-axis-in-matplotlib-with-different-scales
    # smonek's answer
    group = train_smp[['click_hour', 'is_attributed']].groupby(['click_hour'], as_index=False).mean()
    x = group['click_hour']
    ymean = group['is_attributed']
    group = train_smp[['click_hour', 'is_attributed']].groupby(['click_hour'], as_index=False).count()
    ycount = group['is_attributed']

    fig = plt.figure()
    host = fig.add_subplot(111)

    par1 = host.twinx()
    host.set_xlabel("Time")
    host.set_ylabel("Proportion Converted")
    par1.set_ylabel("Click Count")
    # color1 = plt.cm.viridis(0)
    # color2 = plt.cm.viridis(0.5)
    color1 = '#75a1a6'
    color2 = '#a675a1'
    p1, = host.plot(x, ymean, color=color1, label="Proportion Converted")
    p2, = par1.plot(x, ycount, color=color2, label="Click Count")

    lns = [p1, p2]
    host.legend(handles=lns, loc='best')

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())

    plt.savefig("pyplot_multiple_y-axis.png", bbox_inches='tight')

    # beautiful
    import seaborn as sns

    sns.barplot('click_hour', 'is_attributed', data=train_smp)
    plt.title('HOURLY CONVERSION RATIO')
    plt.ylabel('Converted Ratio')


def feature_e_test_1():
    ### 1. Loading data
    # Load subset of the training data
    X_train = pd.read_csv(org_path + os.sep + 'train.csv', nrows=1000000, parse_dates=['click_time'])

    ### 2. Creating Features
    ## 2.1 Extracting time information
    X_train['day'] = X_train['click_time'].dt.day
    X_train['hour'] = X_train['click_time'].dt.hour
    X_train['minute'] = X_train['click_time'].dt.minute
    X_train['second'] = X_train['click_time'].dt.second

    ## 2.2. Confidence Rates for is_attributed
    ATTRIBUTION_CATEGORIES = [
        # V1 Features #
        ['ip'], ['app'], ['device'], ['os'], ['channel'],

        # V2 Features #
        ['app', 'channel'],
        ['app', 'os'],
        ['app', 'device'],

        # V3 Features #
        ['channel', 'os'],
        ['channel', 'device'],
        ['os', 'device']
    ]

    # Aggregation function
    def rate_calculation(x):
        """Calculate the attributed rate. Scale by confidence"""
        rate = x.sum() / float(x.count())
        conf = np.min([1, np.log(x.count()) / log_group])
        return rate * conf

    # Find frequency of is_attributed for each unique value in column
    log_group = np.log(100000)  # 1000 views -> 60% confidence, 100 views -> 40% confidence
    freqs = {}
    for cols in ATTRIBUTION_CATEGORIES:
        # New feature name
        new_feature = '_'.join(cols) + '_confRate'

        # Perform the groupby
        group_object = X_train.groupby(cols)

        # Group sizes
        group_sizes = group_object.size()

        print(
            ">> Calculating confidence-weighted rate for: {}.\n"
            "   Saving to: {}. Group Max /Mean / Median / Min: {} / {} / {} / {}".format(
                cols, new_feature,
                group_sizes.max(),
                np.round(group_sizes.mean(), 2),
                np.round(group_sizes.median(), 2),
                group_sizes.min()
            ))

        # Perform the merge
        X_train = X_train.merge(
            group_object['is_attributed']. \
                apply(rate_calculation). \
                reset_index(). \
                rename(
                index=str,
                columns={'is_attributed': new_feature}
            )[cols + [new_feature]],
            on=cols, how='left'
        )

    X_train.head()

    pass
