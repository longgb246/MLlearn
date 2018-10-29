# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : lgb453476610@163.com
  Date    : 2018/10/15
  Usage   :
"""

from __future__ import print_function
import numpy as np
import pandas as pd
import os
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot

prophet_path = r'/Users/longguangbin/Work/scripts/sample_data/prophet-master/examples'


def test_simple():
    # 包含 ds + y 2列
    df = pd.read_csv(prophet_path + os.sep + 'example_wp_log_peyton_manning.csv')

    m = Prophet()
    m.fit(df)

    future = m.make_future_dataframe(periods=365)
    # future.tail()

    forecast = m.predict(future)
    # forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    # forecast.columns

    fig1 = m.plot(forecast)
    fig2 = m.plot_components(forecast)


def test_carrying_capacity():
    df = pd.read_csv(prophet_path + os.sep + 'example_wp_log_R.csv')

    # cap 设置 carrying capacity 承载能力
    # The important things to note are that cap must be specified for every row in the dataframe, and that it does not
    # have to be constant. If the market size is growing, then cap can be an increasing sequence.
    df['cap'] = 8.5

    # The logistic function has an implicit minimum of 0, and will saturate at 0 the same way that it saturates at the
    # capacity. It is possible to also specify a different saturating minimum.
    m = Prophet(growth='logistic')
    m.fit(df)

    future = m.make_future_dataframe(periods=1826)
    future['cap'] = 8.5  # 指定 carrying capacity 承载能力
    fcst = m.predict(future)
    fig = m.plot(fcst)

    df['y'] = 10 - df['y']
    df['cap'] = 6
    # floor 设置下限 saturating minimum, which is specified with a column floor
    df['floor'] = 1.5

    future['cap'] = 6
    future['floor'] = 1.5
    m = Prophet(growth='logistic')
    m.fit(df)
    fcst = m.predict(future)
    fig = m.plot(fcst)


def test_change_point():
    df = pd.read_csv(prophet_path + os.sep + 'example_wp_log_R.csv')

    # 默认 first 80% of the time series in order to have plenty
    # m = Prophet()
    # m = Prophet(changepoint_range=0.9)
    # flexibility 的调整？
    # m = Prophet(changepoint_prior_scale=0.005)
    # Specifying the locations of the changepoints
    m = Prophet(changepoints=['2014-01-01'])

    m.fit(df)

    future = m.make_future_dataframe(periods=365)
    # future.tail()

    forecast = m.predict(future)
    fig = m.plot(forecast)
    a = add_changepoints_to_plot(fig.gca(), m, forecast)


def test_holiday():
    playoffs = pd.DataFrame({
        'holiday': 'playoff',
        'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                              '2010-01-24', '2010-02-07', '2011-01-08',
                              '2013-01-12', '2014-01-12', '2014-01-19',
                              '2014-02-02', '2015-01-11', '2016-01-17',
                              '2016-01-24', '2016-02-07']),
        'lower_window': 0,
        'upper_window': 1,
    })
    superbowls = pd.DataFrame({
        'holiday': 'superbowl',
        'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
        'lower_window': 0,
        'upper_window': 1,
    })
    holidays = pd.concat((playoffs, superbowls))

    df = pd.read_csv(prophet_path + os.sep + 'example_wp_log_peyton_manning.csv')

    m = Prophet(holidays=holidays)
    m.fit(df)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    # forecast[(forecast['playoff'] + forecast['superbowl']).abs() > 0][['ds', 'playoff', 'superbowl']][-10:]

    fig = m.plot_components(forecast)
    m.plot(forecast)
    m.plot_forecast_component(forecast, 'superbowl')
    # Prophet uses a Fourier order of 3 for weekly seasonality and 10 for yearly seasonality.
    m.plot_yearly()

    # 越高越多变
    # N Fourier terms corresponds to 2N variables used for modeling the cycle
    m = Prophet(yearly_seasonality=20).fit(df)
    a = m.plot_yearly()

    # 添加月的周期性
    m = Prophet(weekly_seasonality=False)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    forecast = m.fit(df).predict(future)
    fig = m.plot_components(forecast)

    # holidays_prior_scale. By default this parameter is 10,  holidays are overfitting?
    m = Prophet(holidays=holidays, holidays_prior_scale=0.05).fit(df)
    forecast = m.predict(future)
    # forecast[(forecast['playoff'] + forecast['superbowl']).abs() > 0][['ds', 'playoff', 'superbowl']][-10:]

    # seasonality_prior_scale
    m = Prophet()
    m.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=0.1)

    # Add some func
    def nfl_sunday(ds):
        date = pd.to_datetime(ds)
        if date.weekday() == 6 and (date.month > 8 or date.month < 2):
            return 1
        else:
            return 0

    df['nfl_sunday'] = df['ds'].apply(nfl_sunday)

    m = Prophet()
    m.add_regressor('nfl_sunday')
    m.fit(df)

    future['nfl_sunday'] = future['ds'].apply(nfl_sunday)
    forecast = m.predict(future)
    fig = m.plot_components(forecast)


def test_multiplicative_season():
    df = pd.read_csv(prophet_path + os.sep + 'example_air_passengers.csv')
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(50, freq='MS')
    forecast = m.predict(future)
    # fig = m.plot(forecast)
    m.plot_components(forecast)

    # 使用复合的趋势。但是我理解的是，即使不加也应该也考虑了这个东西
    m = Prophet(seasonality_mode='multiplicative')
    m.fit(df)
    forecast = m.predict(future)
    # fig = m.plot(forecast)
    m.plot_components(forecast)

    m = Prophet(seasonality_mode='multiplicative')
    m.add_seasonality('quarterly', period=91.25, fourier_order=8, mode='additive')
    m.add_regressor('regressor', mode='additive')
    m.fit(df)
    forecast = m.predict(future)
    m.plot(forecast)
    m.plot_components(forecast)


def get_data():
    from dateutil.parser import parse
    data_path = u'/Users/longguangbin/Work/Documents/SAAS/安踏线下/季节性/sample/duanku_sales.xls'
    data = pd.read_excel(data_path)

    data['week_sale'] = data['sku_num_sum'].rolling(window=7).sum()
    data.loc[:5, ['week_sale']] = data[:6]['sku_num_sum'].cumsum()
    data['week_day'] = data['commit_date'].apply(lambda x: parse(x).weekday())
    data_week = data[data['week_day'] == 6]

    data['commit_month'] = data['commit_date'].apply(lambda x: parse(x).strftime('%Y-%m'))
    data_month = data.groupby(['commit_month']).agg({'sku_num_sum': 'sum'}).reset_index()

    df = data.loc[:, ['commit_date', 'sku_num_sum']]
    df.columns = ['ds', 'y']
    df = data_month.loc[:, ['commit_month', 'sku_num_sum']]
    df.columns = ['ds', 'y']
    df = data_week.loc[:, ['commit_date', 'sku_num_sum']]
    df.columns = ['ds', 'y']
    # df = data_month['sku_num_sum'].values
    # df = data_week['sku_num_sum'].values

    m = Prophet(growth='logistic', changepoint_prior_scale=0.005, changepoint_range=0.9)
    df['cap'] = np.max(df['y']) * 1.2
    df['floor'] = 0
    m.fit(df)
    future = m.make_future_dataframe(periods=360)
    future['cap'] = np.max(df['y']) * 1.2
    future['floor'] = 0
    forecast = m.predict(future)
    fig = m.plot(forecast)
    a = add_changepoints_to_plot(fig.gca(), m, forecast)
    m.plot_components(forecast)
