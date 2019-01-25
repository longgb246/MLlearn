# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/24
"""  
Usage Of 'seasonal_t.py' : 
"""

from __future__ import print_function
import numpy as np
import statsmodels.api as sm

from statsmodels.tsa.filters._utils import (_maybe_get_pandas_wrapper_freq,
                                            _maybe_get_pandas_wrapper)

tmp_t = [480.9434, 230.52441, 504.7211, 540.8153, 501.627,
         409.1944, 374.37903, 489.53506, 312.67752, 291.19086,
         470.88724, 177.3592, 314.66647, 321.39944, 442.20935,
         341.35025, 388.26584, 301.63495, 368.0727, 501.81525,
         408.08752, 463.2604, 371.266, 537.45026, 342.91742,
         204.07265, 543.9972, 547.52484, 408.70642, 323.14615,
         422.0337, 211.76036, 554.76416, 474.24182, 203.53772,
         163.83145, 536.9149, 332.18216, 379.05774, 307.28018,
         231.88252, 517.3504, 215.37202, 354.2208, 128.83694,
         453.93088, 145.72485, 304.53574, 123.102585, 210.59601,
         399.54114, 552.864, 487.92975, 211.01578, 416.77902,
         136.4255, 210.14542, 406.7575, 258.79123, 453.64218,
         565.93164, 437.50073, 489.42966, 487.96707, 311.75174,
         505.17657, 385.06625, 437.62677, 503.72824, 309.22485,
         143.9296, 504.23013, 461.2881, 523.5238, 425.07605,
         502.31516, 424.87448, 349.14236, 503.63968, 302.55377,
         404.2226, 336.10028, 301.12167, 382.11884, 217.45279,
         280.91458, 558.36633, 502.854, 280.77252, 552.22296,
         282.15582, 353.22586, 231.28992, 400.9652, 389.70343,
         184.29485, 525.2271, 239.63177, 349.92728, 345.54123,
         494.46457, 472.31165, 506.7303, 293.048, 539.5341,
         591.973, 482.3038, 276.1311, 517.2454, 235.70436,
         191.2603, 317.9334, 415.91956, 132.51715, 470.67798,
         269.3496, 463.3289, 483.3555, 394.70425, 481.1852,
         165.01405, 404.79242, 437.04517, 240.8491, 344.5829,
         227.59856, 185.56306, 553.403, 492.0085, 592.89374,
         383.91806, 484.10693, 452.75415, 355.32303, 290.3437,
         362.86792, 219.32622, 204.70612, 330.56467, 366.48215,
         449.79123, 108.24265, 159.1035, 433.86963, 555.49084,
         150.75246, 493.467, 279.40707, 455.3822, 471.02298,
         469.03995, 443.409, 507.99814, 261.79694, 204.67578,
         365.05914, 402.5748, 264.5895, 449.6466, 339.0233,
         500.43457, 392.05264, 366.9022, 384.39072, 479.38278,
         419.68665, 325.38818, 197.81471, 308.06293, 440.1154,
         391.71417, 520.26733, 466.71182, 322.11002, 163.78058,
         459.4172, 280.74188, 325.58142, 498.4852, 461.51398,
         368.4209, 544.47577, 144.70995, 275.92233, 391.97507,
         181.75613, 288.6996, 167.1316, 219.26572, 306.16953]
x = np.array(tmp_t)

result = sm.tsa.seasonal_decompose(x, model='multiplicative', freq=7)  # additive#multiplicative

len(x)
len(result.resid)
len(result.trend)
len(result.seasonal)

len(result.resid * result.trend * result.seasonal)

print((result.resid * result.trend * result.seasonal - x) < 10e-8)

# convolution 的计算
from scipy import signal

a = np.array([1, 1, 4, 5])
b = np.array([3, 4])
signal.convolve(a, b, mode='valid')
signal.convolve(b, a, mode='valid')
# res = [4 * 1 + 3 * 1,
#        4 * 1 + 3 * 4,
#        4 * 4 + 3 * 5,]
# c = [3, 4, 5, 6, 7, 74, 5]
# c[0::2]  # 2 为周期，每第0个取出来

# seasonal_decompose 的方法
# 1、使用平滑的方法得出趋势 trend ： seasonal_decompose 使用 convolution 的计算方法得出，即为移动平均计算
# 2、剔除 trend ： data_t  =  data - trend  or  data / trend
# 3、求出周期 seasonal ： data_t 每 freq（指定）求出均值
# 4、剔除 seasonal 得到残差 res ： res  =  data_t - seasonal  or  data_t / seasonal

# 寻找数据的周期性 - 自相关 / FFT（ 快速傅里叶变换 ）

import matplotlib.pyplot as plt
import numpy as np

x = np.random.rand(100)  # create 100 random numbers of which we want the fourier transform
x = x - np.mean(x)  # make sure the average is zero, so we don't get a huge DC offset.
dt = 0.1  # [s] 1/the sampling rate
fftx = np.fft.fft(x)  # the frequency transformed part
# now discard anything  that we do not need..
fftx = fftx[range(int(len(fftx) / 2))]
# now create the frequency axis: it runs from 0 to the sampling rate /2
freq_fftx = np.linspace(0, 2 / dt, len(fftx))
# and plot a power spectrum
plt.plot(freq_fftx, abs(fftx) ** 2)
