#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/19 9:11
# @Author  : zhengdepeng
# @concat  : zhengdepeng@xxx.com
# @File    : crost.py.py
# @Software: PyCharm

"""
description:function to forecast intermittent demand or slow moving items series / this function
Reference :R version package,http://kourentzes.com/forecasting/2014/06/23/intermittent-demand-forecasting-package-for-r/
"""
import numpy as np
import warnings
from scipy.optimize import minimize
import functools


def tsb(data, horizon=10, smoothing_para=None, init_method=("mean", "mean"), num_para=1, cost="mse",
        init_opt=True, opt_on=True):
    """

    :param data:                Intermittent demand time series. ->single time series
    :param horizon:             Forecast horizon.
    :param smoothing_para:      Smoothing parameters. If w == NULL then parameters are optimised.If w is a single
                                parameter then the same is used for smoothing both the demand and the intervals. If two
                                parameters are provided then the second is used to smooth the intervals.
    :param init_method:         Initial values for demand and intervals. This is:
                                c(z,x)  - Vector of two scalars, where first is initial demand and
                                           second is initial interval;
                                "naive" - Initial demand is first non-zero demand and initial interval
                                           is first interval;
                                "mean"  - Same as "naive", but initial interval is the mean of all
                                           in sample intervals.
    :param num_para:            Specifies the number of model parameters. Used only if they are optimised.
                                    1 - Demand and interval parameters are the same
                                    2 - Different demand and interval parameters
    :param cost:                Cost function used for optimisation
                                     "mar" - Mean absolute rate
                                     "msr" - Mean squared rate
                                     "mae" - Mean absolute error
                                     "mse" - Mean squared error
    :param init_opt:            If init.opt==TRUE then initial values are optimised.
    :param opt_on:              This is meant to use only by the optimisation function. When opt.on is
                #               TRUE then no checks on inputs are performed.

    :return:
       model                    Type of model fitted.
       frc_in                   In-sample demand rate.
       frc_out                  Out-of-sample demand rate.
       weights                  Smoothing parameters for demand and interval.
       initial                  Initialisation values for demand and interval smoothing.
       component                List of c.in and c.out containing the non-zero demand and interval vectors for
                                in- and out-of-sample respectively. Third element is the coefficient used to scale
                                demand rate for sba and sbj.
    """
    # trans data model_type to np.ndarray(1)
    data_type = type(data)
    if isinstance(data, np.ndarray):
        data = data.ravel().copy()
    else:
        if isinstance(data, list):
            data = np.array(data).ravel().copy()
        else:
            raise ValueError("data can be either np.ndarray or list. {0} is not allowed".format(data_type))

    # make sure that num_para is of correct length
    if (num_para > 2 or num_para < 1):
        num_para = 2
        warnings.warn("num_para can be either 1 or 2. Overwritten to 2")

    n = data.shape[0]

    # check number of non-zero values -- need to have as least two

    # if ((data!=0).sum())<2:
    #     raise ValueError("need as least two non-zero valus to model time series")

    # TSB decomposition
    # print(data)
    p = (data != 0).astype(int)
    z = data[data != 0]
    # initialize
    if init_method == ("mean", "mean"):
        init = [z[0], p.mean()]
    else:
        init = [z[0], p[0]]

    # optimize parameters if requested
    if opt_on == False:
        if (init_opt == False and smoothing_para is None):
            # print("###################optimizing##################")
            opt_para = crost_opt(data, cost, smoothing_para, num_para, init, init_opt)
            smoothing_para = opt_para[0]
            # init = opt_para[1]
    zfit = np.zeros(n)
    pfit = np.zeros(n)
    if len(smoothing_para == 1):
        smoothing_demand = smoothing_para[0]
        smoothing_prob = smoothing_para[0]
    else:
        smoothing_demand = smoothing_para[0]
        smoothing_prob = smoothing_para[1]
    # assign initial value and parameters
    zfit[0] = init[0]
    pfit[0] = init[1]
    # fit model
    # print(nzd,x,xfit)
    for day in range(1, n):
        pfit[day] = pfit[day - 1] + smoothing_prob * (p[day] - pfit[day - 1])
        if (p[day] == 0):
            zfit[day] = zfit[day - 1]
        else:
            zfit[day] = zfit[day - 1] + smoothing_demand * (data[day] - zfit[day - 1])
    yfit = pfit * zfit

    # calculate in-sample demand
    frc_in = np.append([0], yfit[:-1])
    # forcast out-of-sample demand
    frc_out = np.tile(frc_in[-1], horizon)
    # print("frc_in",frc_in)
    return [frc_in, frc_out, zfit, pfit, smoothing_demand, smoothing_prob]


def crost_opt(data, cost, smoothing_para, num_para, init, init_opt):
    """

    :param data:                Intermittent demand time series. ->single time series
    :param model_type:                Croston's method variant:
                                                1 - "croston" Croston's method;
                                                2 - "sba" Syntetos-Boylan approximation;
                                                3 - "sbj" Shale-Boylan-Johnston.
    :param cost:                Cost function used for optimisation
                                     "mar" - Mean absolute rate
                                     "msr" - Mean squared rate
                                     "mae" - Mean absolute error
                                     "mse" - Mean squared error
    :param smoothing_para:      Smoothing parameters. If w == NULL then parameters are optimised.If w is a single
                                parameter then the same is used for smoothing both the demand and the intervals. If two
                                parameters are provided then the second is used to smooth the intervals.
    :param num_para:            Specifies the number of model parameters. Used only if they are optimised.
                                    1 - Demand and interval parameters are the same
                                    2 - Different demand and interval parameters
    :param init:                initialized estimator point
    :param init_opt:            If init.opt==TRUE then initial values are optimised.
    :return:
        optimized_parameter:    list[smoothing_para,optimized_init]
    """

    if (smoothing_para is None and init_opt == False):
        starting_para = [0.05] * num_para
        bounds = [(0, 1)] * num_para
        lbound = [0] * num_para
        ubound = [1] * num_para
        optimized_para = minimize(
            fun=functools.partial(crost_cost, data=data, smoothing_opt=True, cost=cost, num_para=num_para, init=init,
                                  init_opt=init_opt), x0=starting_para, method="Nelder-Mead", bounds=bounds)['x']
    else:
        raise ValueError("only smoothing_para optimization is supported,you have to set init_opt=False")
    return [optimized_para]


def crost_cost(model_para, smoothing_opt, data, cost, num_para, init, init_opt):
    """
    calculate total cost with given loss function and data
    :param model_para:              the parameter to optimize
    :param data:                    Intermittent demand time series. ->single time series
    :smoothing_para:                Smoothing parameters. If w == NULL then parameters are optimised.If w is a single
                                    parameter then the same is used for smoothing both the demand and the intervals. If two
                                    parameters are provided then the second is used to smooth the intervals.
    :param cost:                    Cost function used for optimisation
                                     "mar" - Mean absolute rate
                                     "msr" - Mean squared rate
                                     "mae" - Mean absolute error
                                     "mse" - Mean squared error
    :param model_type:                    Croston's method variant:
                                                1 - "croston" Croston's method;
                                                2 - "sba" Syntetos-Boylan approximation;
                                                3 - "sbj" Shale-Boylan-Johnston.
    :param num_para:                Specifies the number of model parameters. Used only if they are optimised.
                                    1 - Demand and interval parameters are the same
                                    2 - Different demand and interval parameters
    :param opt_on:                  This is meant to use only by the optimisation function. When opt.on is
                                    TRUE then no checks on inputs are performed.
    :param init_method:             Initial values for demand and intervals. This is:
                                        c(z,x)  - Vector of two scalars, where first is initial demand and
                                                   second is initial interval;
                                        "naive" - Initial demand is first non-zero demand and initial interval
                                                   is first interval;
                                        "mean"  - Same as "naive", but initial interval is the mean of all
                                                   in sample intervals.
    :param init_opt:                If init.opt==TRUE then initial values are optimised.
    :param lbound:                  lower bound for optimized parameter
    :param ubound:                  upper bound for optimized parameter
    :return:
        scalar total cost
    """
    if (smoothing_opt == True and init_opt == False):
        frc_in = tsb(data=data, horizon=0, smoothing_para=model_para, init_method=init, opt_on=True)[0]
        # print(frc_in)
        # print(data)
    else:
        raise ValueError("only smoothing_para optimization is supported,you have to set init_opt=False")
    if cost == 'mse':
        E = data - frc_in
        E = (E ** 2).mean()
    if cost == 'mae':
        E = data - frc_in
        E = np.abs(E).mean()
    if cost == 'mapd':
        E = np.abs(data.sum() - frc_in.sum()) / data.sum()
    if cost not in ['mse', 'mae', 'mapd']:
        raise ValueError("Cost '{cost}' is not supported til now".format(cost=cost))
    # print(E)
    return E


# def test_cost(para,method):
#     data=np.array([1,2])
#     data2 = data * para
#     data3 = data * para
#     if method==1:
#         residual = (data  - data2  -3)**2
#     else:
#         residual = data ** 2 - data2 - data3 * 2
#     return residual.sum()
#
# test_cost2 = functools.partial(test_cost, method=1)
#
# def func1(para):
#     return (para-1)**2 - 3*para
#
#
# minimize(test_cost2,x0=(1),method="Nelder-Mead",tol=1e-2,options={'disp':True})


if __name__ == '__main__':
    data = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    results = tsb(data=data, horizon=7, smoothing_para=None, init_method='naive', num_para=2, cost='mapd',
                  init_opt=False, opt_on=False)
    print(results[1])
    frc_in = results[0]
    print(frc_in)
    print(frc_in.sum())
    print(data.sum())
