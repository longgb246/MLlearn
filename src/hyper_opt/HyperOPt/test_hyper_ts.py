# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2018/12/3
"""  
Usage Of 'test_hyper_ts' :
"""
from __future__ import print_function
import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA

# dta = sm.datasets.sunspots.load_pandas().data
# dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
# del dta["YEAR"]

a_pd = pd.DataFrame(range(10), columns=['value'])
arma_mod = sm.tsa.ARMA(a_pd, order=(0, 2)).fit(disp=False, trend='nc')
print(arma_mod.params)

a = range(10)
order = (0, 1)
arma_mod2 = ARMA(a_pd, order=order).fit(disp=False, trend='nc')
print(arma_mod2.params)
# predict_sunspots = arma_mod2.predict(0, 12)
arma_mod2.forecast(1)


# predict_sunspots = arma_mod.predict('1990', '2012', dynamic=True)
# print(predict_sunspots)


def proper_model(data_ts, maxLag):
    init_bic = float("inf")
    init_p = 0
    init_q = 0
    init_properModel = None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            model = ARMA(data_ts, order=(p, q))
            try:
                results_ARMA = model.fit(disp=-1, method='css')
            except:
                continue
            bic = results_ARMA.bic
            if bic < init_bic:
                init_p = p
                init_q = q
                init_properModel = results_ARMA
                init_bic = bic
    return init_bic, init_p, init_q, init_properModel


# 参数搜索
# import statsmodels.tsa.stattools as st
#
# order = st.arma_order_select_ic(timeseries, max_ar=5, max_ma=5, ic=['aic', 'bic', 'hqic'])
# order.bic_min_order

# for t in range(len(test)):
#     model = ARIMA(history, order=(5, 1, 0))
#     model_fit = model.fit(disp=0)
#     output = model_fit.forecast()
#     yhat = output[0]
#     predictions.append(yhat)
#     obs = test[t]
#     history.append(obs)
#     print('predicted=%f, expected=%f' % (yhat, obs))
#
# # this is the nsteps ahead predictor function

#
# res = sm.tsa.ARMA(y, (3, 2)).fit(trend="nc")

# get what you need for predicting one-step ahead
# params = res.params
# residuals = res.resid
# p = res.k_ar
# q = res.k_ma
# k_exog = res.k_exog
# k_trend = res.k_trend
# steps = 1
#


type(arma_mod.params)
params = arma_mod.params
params['const'] = 0
params['ma.L1.value'] = 1
steps = 2
residuals = arma_mod.resid
p = arma_mod.k_ar
q = arma_mod.k_ma
k_exog = arma_mod.k_exog
k_trend = arma_mod.k_trend
y = a_pd

from statsmodels.tsa.arima_model import _arma_predict_out_of_sample

_arma_predict_out_of_sample(params, steps, residuals, p, q, k_trend, k_exog, endog=y, exog=None, start=len(y))

# -------------------------- Example --------------------------

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.graphics.api import qqplot


def get_data():
    dta_v = sm.datasets.sunspots.load_pandas().data
    dta_v.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
    del dta_v["YEAR"]
    return dta_v


dta = get_data()

# 画时序图
dta.plot(figsize=(12, 8))

# 画 acf、pacf 图
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)

arma_mod20 = sm.tsa.ARMA(dta, (2, 0)).fit(disp=False)
print(arma_mod20.params)
arma_mod30 = sm.tsa.ARMA(dta, (3, 0)).fit(disp=False)
print(arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic)
print(arma_mod30.params)
print(arma_mod30.aic, arma_mod30.bic, arma_mod30.hqic)
sm.stats.durbin_watson(arma_mod30.resid.values)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax = arma_mod30.resid.plot(ax=ax)

resid = arma_mod30.resid
stats.normaltest(resid)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)

r, q, p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
data = np.c_[range(1, 41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))

predict_sunspots = arma_mod30.predict('1990', '2012', dynamic=True)
print(predict_sunspots)

fig, ax = plt.subplots(figsize=(12, 8))
ax = dta.loc['1950':].plot(ax=ax)
fig = arma_mod30.plot_predict('1990', '2012', dynamic=True, ax=ax, plot_insample=False)


def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()


mean_forecast_err(dta.SUNACTIVITY, predict_sunspots)

# Exercise: Can you obtain a better fit for the Sunspots model? (Hint: sm.tsa.AR has a method select_order)
# Simulated ARMA(4,1): Model Identification is Difficult

from statsmodels.tsa.arima_process import arma_generate_sample, ArmaProcess

np.random.seed(1234)
# include zero-th lag
arparams = np.array([1, .75, -.65, -.55, .9])
maparams = np.array([1, .65])

arma_t = ArmaProcess(arparams, maparams)
arma_t.isinvertible
arma_t.isstationary
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.plot(arma_t.generate_sample(nsample=50))

arparams = np.array([1, .35, -.15, .55, .1])
maparams = np.array([1, .65])
arma_t = ArmaProcess(arparams, maparams)
arma_t.isstationary

arma_rvs = arma_t.generate_sample(nsample=500, burnin=250, scale=2.5)

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(arma_rvs, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(arma_rvs, lags=40, ax=ax2)
# For mixed ARMA processes the Autocorrelation function is a mixture of exponentials and damped sine waves after (q-p) lags.
# The partial autocorrelation function is a mixture of exponentials and dampened sine waves after (p-q) lags.

arma11 = sm.tsa.ARMA(arma_rvs, (1, 1)).fit(disp=False)
resid = arma11.resid
r, q, p = sm.tsa.acf(resid, qstat=True)
data = np.c_[range(1, 41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))

arma41 = sm.tsa.ARMA(arma_rvs, (4, 1)).fit(disp=False)
resid = arma41.resid
r, q, p = sm.tsa.acf(resid, qstat=True)
data = np.c_[range(1, 41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))

# Exercise: How good of in-sample prediction can you do for another series, say, CPI
macrodta = sm.datasets.macrodata.load_pandas().data
macrodta.index = pd.Index(sm.tsa.datetools.dates_from_range('1959Q1', '2009Q3'))
cpi = macrodta["cpi"]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax = cpi.plot(ax=ax)
ax.legend()
# P-value of the unit-root test, resoundly rejects the null of no unit-root.

print(sm.tsa.adfuller(cpi)[1])
