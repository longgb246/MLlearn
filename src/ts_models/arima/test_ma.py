# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2018/12/12
"""  
Usage Of 'test_ma' : 
"""

from statsmodels.tsa.arima_model import _arma_predict_out_of_sample


def wma(data, p, w=None, step=1):
    """ Use the data series to calculate the wma series.

    :param list data: ts data
    :param int p: p parameter of MA, use the length of data (from right)
    :param list w: weight of WMA
    :param int step: predict step
    :return: the predict of wma
    """
    w = w[::-1] if isinstance(w, list) else [1.0 / p] * int(p)
    residuals = [0]
    q = 0
    k_exog = 0
    k_trend = 0
    res = _arma_predict_out_of_sample(w, step, residuals, p, q, k_trend, k_exog, endog=data)
    return res


def test_wma():
    data = range(10)
    p = 2
    w = [0.3, 0.2]
    wma(data, p, w=w, step=3)


if __name__ == '__main__':
    test_wma()
