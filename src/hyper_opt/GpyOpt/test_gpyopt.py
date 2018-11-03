# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : lgb453476610@163.com
  Date    : 2018/10/29
  Usage   :
"""

from __future__ import print_function
import sys
import platform
import time
import math
import datetime
from dateutil.parser import parse
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# ----- Necessary -----
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

# ----- Matplotlib -----
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

pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 180)  # 150
pd.set_option('display.max_columns', 40)

# ============================ GpyOpt Learn ============================

import GPy
import GPyOpt
from GPyOpt.experiment_design import initial_design
from numpy.random import seed


# no
def example_reference_manual():
    # ------------- Learn - 1 -------------
    def myf(x):
        print(x)
        return (2 * x) ** 2

    # defines the name, type and domain of the variables.
    bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (-1, 1)}]
    max_iter = 15

    myProblem = GPyOpt.methods.BayesianOptimization(myf, bounds)
    myProblem.run_optimization(max_iter)

    print(myProblem.x_opt)  # x
    print(myProblem.fx_opt)  # y

    # ------------- Learn - 2 -------------
    f_true = GPyOpt.objective_examples.experiments1d.forrester()  # noisy version
    bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0, 1)}]  # problem constraints

    f_true.plot()
    seed(123)

    # Creates GPyOpt object with the model and anquisition function
    myBopt = GPyOpt.methods.BayesianOptimization(f=f_true.f,  # function to optimize
                                                 domain=bounds,  # box-constraints of the problem
                                                 acquisition_type='EI',
                                                 exact_feval=True)  # Selects the Expected improvement
    # Run the optimization
    max_iter = 15  # evaluation budget
    max_time = 60  # time budget
    eps = 10e-6  # Minimum allows distance between the las two observations

    myBopt.run_optimization(max_iter, max_time, eps)
    myBopt.plot_acquisition()
    myBopt.plot_convergence()

    # ------------- Learn - 3 -------------
    plt.style.use('classic')
    f_true = GPyOpt.objective_examples.experiments2d.sixhumpcamel()
    f_sim = GPyOpt.objective_examples.experiments2d.sixhumpcamel(sd=0.1)
    bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': f_true.bounds[0]},
              {'name': 'var_2', 'type': 'continuous', 'domain': f_true.bounds[1]}]
    f_true.plot()

    # Creates three identical objects that we will later use to compare the optimization strategies
    myBopt2D = GPyOpt.methods.BayesianOptimization(f_sim.f,
                                                   domain=bounds,
                                                   model_type='GP',
                                                   acquisition_type='EI',
                                                   normalize_Y=True,
                                                   acquisition_weight=2)

    # runs the optimization for the three methods
    max_iter = 40  # maximum time 40 iterations
    max_time = 60  # maximum time 60 seconds

    myBopt2D.run_optimization(max_iter, max_time, verbosity=False)

    myBopt2D.plot_acquisition()
    myBopt2D.plot_convergence()


# no
def example_scikitlearn():
    import GPy
    import GPyOpt
    import numpy as np
    from numpy.random import seed

    seed(12345)
    from sklearn import svm

    # ------------------ 简单的使用 svr 做预测 ------------------
    GPy.util.datasets.authorize_download = lambda x: True  # prevents requesting authorization for download.
    data = GPy.util.datasets.olympic_marathon_men()
    X = data['X']
    Y = data['Y']
    X_train = X[:20]
    Y_train = Y[:20, 0]
    X_test = X[20:]
    Y_test = Y[20:, 0]

    svr = svm.SVR()
    svr.fit(X_train, Y_train)
    Y_train_pred = svr.predict(X_train)
    Y_test_pred = svr.predict(X_test)
    print("The default parameters obtained: C=" + str(svr.C) + ", epilson=" +
          str(svr.epsilon) + ", gamma=" + str(svr.gamma))

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    ax.plot(X_train, Y_train_pred, 'b', label='pred-train', alpha=0.75)
    ax.plot(X_test, Y_test_pred, 'g', label='pred-test', alpha=0.75)
    ax.plot(X_train, Y_train, 'rX', label='ground truth', alpha=0.75)
    ax.plot(X_test, Y_test, 'rX', alpha=0.75)
    ax.legend(loc='best')
    plt.show()
    print("RMSE = " + str(np.sqrt(np.square(Y_test_pred - Y_test).mean())))

    # ------------------ 对 svr 调参 ------------------
    nfold = 3

    def fit_svr_val(x):
        x = np.atleast_2d(np.exp(x))
        fs = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            fs[i] = 0
            for n in range(nfold):
                idx = np.array(range(X_train.shape[0]))
                idx_valid = np.logical_and(idx >= X_train.shape[0] / nfold * n,
                                           idx < X_train.shape[0] / nfold * (n + 1))
                idx_train = np.logical_not(idx_valid)
                svr = svm.SVR(C=x[i, 0], epsilon=x[i, 1], gamma=x[i, 2])
                svr.fit(X_train[idx_train], Y_train[idx_train])
                fs[i] += np.sqrt(np.square(svr.predict(X_train[idx_valid]) - Y_train[idx_valid]).mean())
            fs[i] *= 1. / nfold
        return fs

    domain = [{'name': 'C', 'type': 'continuous', 'domain': (0., 7.)},
              {'name': 'epsilon', 'type': 'continuous', 'domain': (-12., -2.)},
              {'name': 'gamma', 'type': 'continuous', 'domain': (-12., -2.)}]

    opt = GPyOpt.methods.BayesianOptimization(f=fit_svr_val,  # function to optimize
                                              domain=domain,  # box-constraints of the problem
                                              acquisition_type='LCB',  # LCB acquisition
                                              acquisition_weight=0.1)  # Exploration exploitation
    opt.run_optimization(max_iter=50)
    opt.plot_convergence()

    x_best = np.exp(opt.X[np.argmin(opt.Y)])
    print("The best parameters obtained: C=" + str(x_best[0]) + ", epilson=" + str(x_best[1]) +
          ", gamma=" + str(x_best[2]))
    svr = svm.SVR(C=x_best[0], epsilon=x_best[1], gamma=x_best[2])
    svr.fit(X_train, Y_train)
    Y_train_pred = svr.predict(X_train)
    Y_test_pred = svr.predict(X_test)

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    ax.plot(X_train, Y_train_pred, 'b', label='pred-train', alpha=0.75)
    ax.plot(X_test, Y_test_pred, 'g', label='pred-test', alpha=0.75)
    ax.plot(X_train, Y_train, 'rX', label='ground truth', alpha=0.75)
    ax.plot(X_test, Y_test, 'rX', alpha=0.75)
    ax.legend(loc='best')
    plt.show()
    print("RMSE = " + str(np.sqrt(np.square(Y_test_pred - Y_test).mean())))


# no
def example_initial_design():
    func = GPyOpt.objective_examples.experimentsNd.alpine1(input_dim=2)

    mixed_domain = [{'name': 'var1_2', 'type': 'continuous', 'domain': (-10, 10), 'dimensionality': 1},
                    {'name': 'var5', 'type': 'continuous', 'domain': (-1, 5)}]

    space = GPyOpt.Design_space(mixed_domain)
    data_init = 500

    ### --- Grid design
    X = initial_design('grid', space, data_init)
    plt.plot(X[:, 0], X[:, 1], 'b.')
    plt.title('Grid design')

    ### --- Random initial design
    X = initial_design('random', space, data_init)
    plt.plot(X[:, 0], X[:, 1], 'b.')
    plt.title('Random design')

    ### --- Latin design
    X = initial_design('latin', space, data_init)
    plt.plot(X[:, 0], X[:, 1], 'b.')
    plt.title('Latin design')

    ### --- Sobol design
    X = initial_design('sobol', space, data_init)
    plt.plot(X[:, 0], X[:, 1], 'b.')
    plt.title('Sobol design')
    pass
