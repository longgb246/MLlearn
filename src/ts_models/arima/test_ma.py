# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2018/12/12
"""  
Usage Of 'test_ma' : 
"""

from statsmodels.tsa.arima_model import _arma_predict_out_of_sample


def MA_predict(data, p, w=None, step=1):
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
    w = w[::-1] or [1.0 / p] * p
    residuals = [0]
    q = 0
    k_exog = 0
    k_trend = 0
    res = _arma_predict_out_of_sample(w, step, residuals, p, q, k_trend, k_exog, endog=data)
    return res


def test_wma():
    data = range(10)
    p = 2
    w = [0.3, 0.7]
    MA_predict(data, p, w=w, step=3)


if __name__ == '__main__':
    test_wma()
