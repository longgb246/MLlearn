# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2018/12/12
"""  
Usage Of 'test_offline' : 
"""

from __future__ import print_function
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.parser import parse
import datetime
import time

import matplotlib.ticker as ticker
from statsmodels.tsa.arima_model import _arma_predict_out_of_sample

import lgblearn as lg

path = r'/Users/longguangbin/Work/scripts/local_test/anta_offline'


def _split_sku(data):
    data['sku_code'] = data['sku_id'].apply(lambda x: x.split('_')[0])
    data = data.drop(['sku_id'], axis=1)
    return data


# 画图
def plot_pre_real_s(preprocess_data, input_data, pre_data, sku_code, store_id, last_date):
    """ Plot the series fig of pre and real. """
    # 读取 sku_code - store_id
    this_preprocess_data = preprocess_data[
        (preprocess_data['sku_code'] == sku_code) & (preprocess_data['store_id'] == store_id)]
    this_input_data = input_data[
        (input_data['sku_code'] == sku_code) & (input_data['store_id'] == store_id)]
    this_pre_data = pre_data[
        (pre_data['sku_code'] == sku_code) & (pre_data['store_id'] == store_id)]

    this_preprocess_data = this_preprocess_data.loc[:, ['sku_code', 'store_id', 'sale', 'dt']]. \
        rename(columns={'sale': 'filter_sale'}).sort_values('dt')
    this_input_data = this_input_data.loc[:, ['sku_code', 'store_id', 'sale', 'sale_date']]. \
        rename(columns={'sale_date': 'dt'}).sort_values('dt')

    # 预测 list 转 df
    pre_list = eval(this_pre_data['sale_list'].values[0])
    data_list = lg.date_range((parse(last_date) + datetime.timedelta(1)).strftime('%Y-%m-%d'), len(pre_list) - 1)
    this_pre_df = pd.DataFrame(zip(data_list, pre_list), columns=['dt', 'pre_sales'])
    this_pre_df_1 = this_pre_df[:7]

    # 误差
    error_data = this_pre_df_1.merge(this_input_data, on=['dt'], how='left').fillna(0)

    # 画图
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111)
    ax.plot(this_preprocess_data['dt'], this_preprocess_data['filter_sale'], label='filter_sale')
    ax.plot(this_pre_df_1['dt'], this_pre_df_1['pre_sales'], label='pre_sale')
    ax.plot(this_input_data['dt'], this_input_data['sale'], marker='.', linestyle='', markerfacecolor='r',
            label='real_sale')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 0.55))
    ax.set_title('SKU : {sku_code}  ,  STORE_ID : {store_id}'.format(sku_code=sku_code, store_id=store_id))
    for it in error_data.loc[:, ['dt', 'sale', 'pre_sales']].values:
        ax.plot([it[0], it[0]], [min(it[1], it[2]), max(it[1], it[2])], linestyle='--', linewidth=0.85, color='#C44E52')
    # plt.xticks(rotation=45)
    plt.tight_layout(rect=(0.06, 0, 0.98, 0.95))
    tick_num = 15  # 刻度数目
    x_tick_labels = list(ax.get_xticklabels())
    tick_spacing = int(np.ceil(len(x_tick_labels) * 1.0 / tick_num))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    # plt.subplots_adjust(right=0.85)
    return fig, ax


# 计算mape
def _cal_mape(pre_data, input_data, last_date):
    """ pre_data : 'sku_code', 'store_id', 'pre_sale'.  pre_sale is the sum of x days """
    date1 = (parse(last_date) + datetime.timedelta(1)).strftime('%Y-%m-%d')
    date2 = (parse(last_date) + datetime.timedelta(7)).strftime('%Y-%m-%d')
    print(date1, date2)
    tmp_data = input_data[(input_data['sale_date'] >= date1) & (input_data['sale_date'] <= date2)]. \
        groupby(['sku_code', 'store_id']).agg({'sale': 'sum'}).reset_index()
    all_data = pre_data.loc[:, ['sku_code', 'store_id', 'pre_sale']]. \
        merge(tmp_data, on=['sku_code', 'store_id'], how='left').fillna(0)
    all_data['gap'] = (all_data['sale'] - all_data['pre_sale']).abs()
    mape = all_data['gap'].sum() / all_data['sale'].sum()
    return mape, all_data


# 计算mape
def cal_mape(pre_data, input_data, last_date):
    """ Calculate the mape. """
    pre_data['pre_sale'] = pre_data['sale_list'].apply(lambda x: sum(eval(x)[:7]))
    mape, all_data = _cal_mape(pre_data, input_data, last_date)
    return mape, all_data


# ma 预测方法
def MA_predict(data, p=2, w=None, step=1):
    """

    :param data: ts data
    :param p: p parameter of MA
    :param w: weight of WMA
    :param step: predict step
    :return:
    """
    # params = [0.5] * order[0]
    # steps = 3
    # residuals = [0]
    # p = order[0]
    # q = order[1]
    # k_exog = 0
    # k_trend = 0
    # y = a
    # _arma_predict_out_of_sample(params, steps, residuals, p, q, k_trend, k_exog, endog=y, exog=None, start=len(y))
    p = min(len(data), p)
    w = w[::-1] if w is not None else [1.0 / p] * p
    residuals = [0]
    q = 0
    k_exog = 0
    k_trend = 0
    res = _arma_predict_out_of_sample(w, step, residuals, p, q, k_trend, k_exog, endog=data)
    return res


def test_ma_predict():
    data = range(10)
    p = 7
    # w = [0.3, 0.7]
    # MA_predict(data, p, w=w, step=3)
    MA_predict(data, p, step=7)


last_date = '2018-11-28'

# ------------------------------------------------
input_data = pd.read_table(path + os.sep + 'input_data.tsv')
preprocess_data = pd.read_table(path + os.sep + 'preprocess_df_11_30.tsv')
# preprocess_data = pd.read_table(path + os.sep + 'preprocess_df_11_28.tsv')
pre_data = pd.read_table(path + os.sep + 'pre_data_11_30.tsv')
# pre_data = pd.read_table(path + os.sep + 'pre_data_11_28.tsv')
preprocess_data = _split_sku(preprocess_data)
# last_date = '2018-11-30'

# ------------------------------------------------
# sku_code, store_id = ['19847302-3', 'KL3A']  # '2018-11-28'
# sku_code, store_id = ['19847321-1', 'K50M']  # '2018-11-28'
# sku_code, store_id = ['19846301-3', 'KL0C']
# sku_code, store_id = ['19847312-2', 'K507']
# sku_code, store_id = ['19847302-3', 'KL30']
sku_code, store_id = ['11741206-3/10.5', 'K53F']

# plot series
plot_pre_real_s(preprocess_data, input_data, pre_data, sku_code, store_id, last_date)
# mape
mape, all_data = cal_mape(pre_data, input_data, last_date)

all_data['nbs_gap'] = all_data['sale'] - all_data['pre_sale']
all_data.sort_values(['gap'], ascending=False)
all_data.sort_values(['nbs_gap'])

all_data.sort_values(['gap', 'sale'], ascending=[True, False])

# ------------------------------------------------
# ------------------------------------------------
# 实际数据
input_data = pd.read_table(path + os.sep + 'input_data.tsv')
# 时序数据
preprocess_data = pd.read_table(path + os.sep + 'preprocess_df_11_28.tsv')
preprocess_data = _split_sku(preprocess_data)
# 训练 - 预测  11-28 的 last 实际
train_df = pd.read_table(path + os.sep + 'train_df_11_28.tsv')
target_df = pd.read_table(path + os.sep + 'target_df_11_28.tsv')

train_x_df = train_df[train_df['sku_id'].apply(lambda x: x.split('$')[0] != 'forecast')]
forecast_x_df = train_df[train_df['sku_id'].apply(lambda x: x.split('$')[0] == 'forecast')]


# 1、上周销量直接出预测
def ma1_pre():
    date1 = (parse(last_date) - datetime.timedelta(6)).strftime('%Y-%m-%d')
    ma1_df = preprocess_data[preprocess_data['dt'] >= date1].groupby(['sku_code', 'store_id']). \
        agg({'sale': 'sum'}).reset_index().rename(columns={'sale': 'pre_sale'})
    mape, all_data = _cal_mape(ma1_df, input_data, last_date)
    # 1.7148555034637185


# 2、使用 ma / wma 计算方法
def ma2_pre():
    date1 = (parse(last_date) - datetime.timedelta(20)).strftime('%Y-%m-%d')
    ma2_df = preprocess_data[preprocess_data['dt'] >= date1]
    # run models
    p = 7
    res = []
    t1 = time.time()
    for i, (key, group) in enumerate(ma2_df.groupby(['sku_code', 'store_id'])):
        if divmod(i, 10000)[1] == 0:
            print(i)
        res.append(list(key) + [float(np.sum(MA_predict(group['sale'].values, p=p, step=7)))])
    res_pd = pd.DataFrame(res, columns=['sku_code', 'store_id', 'pre_sale'])
    lg.run_time(t1)
    # cal mape
    mape, all_data = _cal_mape(res_pd, input_data, last_date)
    # 1.8790014802985942
