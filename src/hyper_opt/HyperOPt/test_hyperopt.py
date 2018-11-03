# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : lgb453476610@163.com
  Date    : 2018/11/1
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
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import hyperopt.pyll.stochastic

from sklearn.datasets import load_iris
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize, scale
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB


def ex_simple_1():
    # ------------------ 简单的例子 - 一次函数 ------------------
    best = fmin(fn=lambda x: x,
                space=hp.uniform('x', 0, 1),
                algo=tpe.suggest,
                max_evals=100)
    print(best)

    # ------------------ 二次函数 ------------------
    best = fmin(
        fn=lambda x: (x - 1) ** 2,
        space=hp.uniform('x', -2, 2),
        algo=tpe.suggest,
        max_evals=100)
    print(best)

    # ------------------ 搜索空间 ------------------
    space = {
        'x': hp.uniform('x', 0, 1),
        'y': hp.normal('y', 0, 1),
        'name': hp.choice('name', ['alice', 'bob']),
    }
    print(hyperopt.pyll.stochastic.sample(space))

    # ------------------ 使用 trails 路径 ------------------
    def f(params):
        x = params['x']
        val = x ** 2
        return {'loss': val, 'status': STATUS_OK}

    fspace = {
        'x': hp.uniform('x', -5, 5)
    }
    trials = Trials()
    best = fmin(fn=f, space=fspace, algo=tpe.suggest, max_evals=50, trials=trials)
    print('best:', best)
    print('trials:')
    for trial in trials.trials[:2]:
        print(trial)

    # x vs t
    xs = [t['tid'] for t in trials.trials]
    ys = [t['misc']['vals']['x'] for t in trials.trials]
    f, ax = plt.subplots(1)
    ax.set_xlim(xs[0] - 10, xs[-1] + 10)
    ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
    ax.set_title('$x$ $vs$ $t$ ', fontsize=18)
    ax.set_xlabel('$t$', fontsize=16)
    ax.set_ylabel('$x$', fontsize=16)

    # val vs x
    f, ax = plt.subplots(1)
    xs = [t['misc']['vals']['x'] for t in trials.trials]
    ys = [t['result']['loss'] for t in trials.trials]
    ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
    ax.set_title('$val$ $vs$ $x$ ', fontsize=18)
    ax.set_xlabel('$x$', fontsize=16)
    ax.set_ylabel('$val$', fontsize=16)

    # ------------------ iris ------------------
    iris = load_iris()
    print(iris.feature_names)  # input names
    print(iris.target_names)  # output names
    print(iris.DESCR)  # everything else
    sns.set(style="whitegrid", palette="husl")
    iris = sns.load_dataset("iris")
    print(iris.head())
    iris = pd.melt(iris, "species", var_name="measurement")
    print(iris.head())
    f, ax = plt.subplots(1, figsize=(15, 10))
    sns.stripplot(x="measurement", y="value", hue="species", data=iris, jitter=True, edgecolor="white", ax=ax)

    # ------------------ knn1 ------------------
    def hyperopt_train_test(params):
        clf = KNeighborsClassifier(**params)
        return cross_val_score(clf, X, y).mean()

    def f1(params):
        acc = hyperopt_train_test(params)
        return {'loss': -acc, 'status': STATUS_OK}

    space4knn = {
        'n_neighbors': hp.choice('n_neighbors', range(1, 50))  # 使用不同范围搜索的结果可能不一样
        # 'n_neighbors': hp.choice('n_neighbors', range(1, 100))
    }
    iris = load_iris()
    X = iris.data
    y = iris.target
    trials = Trials()
    best = fmin(f1, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
    print('best:')
    print(best)
    # plot
    xs = [t['misc']['vals']['n_neighbors'] for t in trials.trials]
    ys = [-t['result']['loss'] for t in trials.trials]
    f, ax = plt.subplots(1)  # , figsize=(10,10))
    ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.5)
    ax.set_title('Iris Dataset - KNN', fontsize=18)
    ax.set_xlabel('n_neighbors', fontsize=12)
    ax.set_ylabel('cross validation accuracy', fontsize=12)

    # ------------------ knn2 ------------------
    def hyperopt_train_test3(params):
        X_ = X[:]
        if 'normalize' in params:
            if params['normalize'] == 1:
                X_ = normalize(X_)
        if 'scale' in params:
            if params['scale'] == 1:
                X_ = scale(X_)
        del params['normalize']
        del params['scale']
        clf = KNeighborsClassifier(**params)
        return cross_val_score(clf, X_, y).mean()

    def f3(params):
        acc = hyperopt_train_test3(params)
        return {'loss': -acc, 'status': STATUS_OK}

    space4knn = {
        'n_neighbors': hp.choice('n_neighbors', range(1, 50)),
        'scale': hp.choice('scale', [0, 1]),
        'normalize': hp.choice('normalize', [0, 1])
    }
    trials = Trials()
    best = fmin(f3, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
    print('best:')
    print(best)
    # plot
    parameters = ['n_neighbors', 'scale', 'normalize']
    cols = len(parameters)
    f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(15, 5))
    cmap = plt.cm.jet
    for i, val in enumerate(parameters):
        xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
        ys = [-t['result']['loss'] for t in trials.trials]
        xs, ys = zip(*sorted(zip(xs, ys)))
        ys = np.array(ys)
        axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75, c=cmap(float(i) / len(parameters)))
        axes[i].set_title(val)

    # ------------------ Support Vector Machines (SVM) ------------------
    def hyperopt_train_test4(params):
        X_ = X[:]
        if 'normalize' in params:
            if params['normalize'] == 1:
                X_ = normalize(X_)
        if 'scale' in params:
            if params['scale'] == 1:
                X_ = scale(X_)
        del params['normalize']
        del params['scale']
        clf = SVC(**params)
        return cross_val_score(clf, X_, y).mean()

    def f4(params):
        acc = hyperopt_train_test4(params)
        return {'loss': -acc, 'status': STATUS_OK}

    space4svm = {
        'C': hp.uniform('C', 0, 20),
        'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
        'gamma': hp.uniform('gamma', 0, 20),
        'scale': hp.choice('scale', [0, 1]),
        'normalize': hp.choice('normalize', [0, 1])
    }
    trials = Trials()
    best = fmin(f4, space4svm, algo=tpe.suggest, max_evals=100, trials=trials)
    print('best:')
    print(best)
    # plot
    parameters = ['C', 'kernel', 'gamma', 'scale', 'normalize']
    cols = len(parameters)
    f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(20, 5))
    cmap = plt.cm.jet
    for i, val in enumerate(parameters):
        xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
        ys = [-t['result']['loss'] for t in trials.trials]
        xs, ys = zip(*sorted(zip(xs, ys)))
        axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.25, c=cmap(float(i) / len(parameters)))
        axes[i].set_title(val)
        axes[i].set_ylim([0.9, 1.0])

    # ------------------ Decision Trees ------------------
    def hyperopt_train_test5(params):
        X_ = X[:]
        if 'normalize' in params:
            if params['normalize'] == 1:
                X_ = normalize(X_)
        if 'scale' in params:
            if params['scale'] == 1:
                X_ = scale(X_)
        del params['normalize']
        del params['scale']
        clf = DecisionTreeClassifier(**params)
        return cross_val_score(clf, X_, y).mean()

    def f5(params):
        acc = hyperopt_train_test5(params)
        return {'loss': -acc, 'status': STATUS_OK}

    space4dt = {
        'max_depth': hp.choice('max_depth', range(1, 20)),
        'max_features': hp.choice('max_features', range(1, 5)),
        'criterion': hp.choice('criterion', ["gini", "entropy"]),
        'scale': hp.choice('scale', [0, 1]),
        'normalize': hp.choice('normalize', [0, 1])
    }
    trials = Trials()
    best = fmin(f5, space4dt, algo=tpe.suggest, max_evals=300, trials=trials)
    print('best:')
    print(best)
    # plot
    parameters = ['max_depth', 'max_features', 'criterion', 'scale', 'normalize']  # decision tree
    cols = len(parameters)
    f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(20, 5))
    cmap = plt.cm.jet
    for i, val in enumerate(parameters):
        xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
        ys = [-t['result']['loss'] for t in trials.trials]
        xs, ys = zip(*sorted(zip(xs, ys)))
        ys = np.array(ys)
        axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.5, c=cmap(float(i) / len(parameters)))
        axes[i].set_title(val)
        # axes[i].set_ylim([0.9,1.0])

    # ------------------ Random Forests ------------------
    def hyperopt_train_test6(params):
        X_ = X[:]
        if 'normalize' in params:
            if params['normalize'] == 1:
                X_ = normalize(X_)
        if 'scale' in params:
            if params['scale'] == 1:
                X_ = scale(X_)
        del params['normalize']
        del params['scale']
        clf = RandomForestClassifier(**params)
        return cross_val_score(clf, X_, y).mean()

    def f6(params):
        global best
        acc = hyperopt_train_test6(params)
        if acc > best:
            best = acc
            print('new best:', best, params)
        return {'loss': -acc, 'status': STATUS_OK}

    space4rf = {
        'max_depth': hp.choice('max_depth', range(1, 20)),
        'max_features': hp.choice('max_features', range(1, 5)),
        'n_estimators': hp.choice('n_estimators', range(1, 20)),
        'criterion': hp.choice('criterion', ["gini", "entropy"]),
        'scale': hp.choice('scale', [0, 1]),
        'normalize': hp.choice('normalize', [0, 1])
    }
    best = 0
    trials = Trials()
    best = fmin(f6, space4rf, algo=tpe.suggest, max_evals=300, trials=trials)
    print('best:')
    print(best)
    # plot
    parameters = ['n_estimators', 'max_depth', 'max_features', 'criterion', 'scale', 'normalize']
    f, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    cmap = plt.cm.jet
    for i, val in enumerate(parameters):
        print(i, val)
        xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
        ys = [-t['result']['loss'] for t in trials.trials]
        xs, ys = zip(*sorted(zip(xs, ys)))
        ys = np.array(ys)
        axes[i / 3, i % 3].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.5, c=cmap(float(i) / len(parameters)))
        axes[i / 3, i % 3].set_title(val)
        # axes[i/3,i%3].set_ylim([0.9,1.0])'

    # ------------------ All Together Now ------------------
    def hyperopt_train_test7(params):
        t = params['type']
        del params['type']
        if t == 'naive_bayes':
            clf = BernoulliNB(**params)
        elif t == 'svm':
            clf = SVC(**params)
        elif t == 'dtree':
            clf = DecisionTreeClassifier(**params)
        elif t == 'knn':
            clf = KNeighborsClassifier(**params)
        else:
            return 0
        return cross_val_score(clf, X, y).mean()

    def f7(params):
        global best, count
        count += 1
        acc = hyperopt_train_test7(params.copy())
        if acc > best:
            print('new best:', acc, 'using', params['type'])
            best = acc
        if count % 50 == 0:
            print('iters:', count, ', acc:', acc, 'using', params)
        return {'loss': -acc, 'status': STATUS_OK}

    space = hp.choice('classifier_type', [
        {
            'type': 'naive_bayes',
            'alpha': hp.uniform('alpha', 0.0, 2.0)
        },
        {
            'type': 'svm',
            'C': hp.uniform('C', 0, 10.0),
            'kernel': hp.choice('kernel', ['linear', 'rbf']),
            'gamma': hp.uniform('gamma', 0, 20.0)
        },
        {
            'type': 'randomforest',
            'max_depth': hp.choice('max_depth', range(1, 20)),
            'max_features': hp.choice('max_features', range(1, 5)),
            'n_estimators': hp.choice('n_estimators', range(1, 20)),
            'criterion': hp.choice('criterion', ["gini", "entropy"]),
            'scale': hp.choice('scale', [0, 1]),
            'normalize': hp.choice('normalize', [0, 1])
        },
        {
            'type': 'knn',
            'n_neighbors': hp.choice('knn_n_neighbors', range(1, 50))
        }
    ])
    count = 0
    best = 0
    trials = Trials()
    best = fmin(f7, space, algo=tpe.suggest, max_evals=1500, trials=trials)
    print('best:')
    print(best)
    pass
