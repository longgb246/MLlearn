# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : lgb453476610@163.com
  Date    : 2018/10/24
  Usage   :
"""

from __future__ import print_function
import warnings

warnings.filterwarnings('ignore')

from pyramid.arima import auto_arima
from pyramid.arima.utils import ndiffs
from pyramid.arima.utils import nsdiffs

import matplotlib.pyplot as plt

plt.style.use('seaborn')


class ArimaModel(object):

    def __init__(self):
        self.model_name = None
        self.model = None
        self.pred = None
        self.org_data = None
        self.pre_len = None

    def autoarima(self, data, pre_len=30):
        D_f = nsdiffs(data, m=3, max_D=5, test='ch')
        d_f = ndiffs(data, alpha=0.05, test='kpss', max_d=5)
        if len(data) <= 30:
            seasonal = False
        else:
            seasonal = True
        try:
            stepwise_fit = auto_arima(data, start_p=0, start_q=0, max_p=3, max_q=3, m=12,
                                      start_P=0, seasonal=seasonal, d=d_f, D=D_f, trace=False,
                                      error_action='ignore',  # don't want to know if an order does not work
                                      suppress_warnings=True,  # don't want convergence warnings
                                      stepwise=True)  # set to stepwise
        except:
            stepwise_fit = auto_arima(data, start_p=0, start_q=0, max_p=3, max_q=0, m=12,
                                      start_P=0, seasonal=False, d=0, D=0, trace=False,
                                      error_action='ignore',  # don't want to know if an order does not work
                                      suppress_warnings=True,  # don't want convergence warnings
                                      stepwise=True)  # set to stepwise
        output = stepwise_fit.predict(n_periods=pre_len)

        self.model_name = 'autoarima'
        self.model = stepwise_fit
        self.pred = output
        self.org_data = data
        self.pre_len = pre_len
        return output

    def plot_pre(self):
        data = self.org_data
        pre_len = self.pre_len
        output = self.pred

        real_index = range(len(data))
        pre_index = range(len(data), len(data) + pre_len)

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.plot(real_index, data)
        ax.plot(pre_index, output)
        plt.show()


if __name__ == '__main__':
    pass
