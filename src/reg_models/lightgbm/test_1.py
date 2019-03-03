# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : lgb453476610@163.com
  Date    : 2018/11/4
  Usage   :
"""

from __future__ import print_function
import sys
import platform

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

import pandas as pd

pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 180)  # 150
pd.set_option('display.max_columns', 40)

# ---------------------- Learn ----------------------
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np
import json
import time
from scipy.special import expit

try:
    import cPickle as pickle
except BaseException:
    import pickle

if lgb.compat.MATPLOTLIB_INSTALLED:
    import matplotlib.pyplot as plt
else:
    raise ImportError('You need to install matplotlib for plot_example.py.')


def test_simple_example():
    print('Loading data...')
    # load or create your dataset
    df_train = pd.read_csv(r'/Users/longguangbin/Work/Codes/MLlearn/src/reg_models/LightGBM/data/regression.train',
                           header=None, sep='\t')
    df_test = pd.read_csv(r'/Users/longguangbin/Work/Codes/MLlearn/src/reg_models/LightGBM/data/regression.test',
                          header=None, sep='\t')

    y_train = df_train[0]
    y_test = df_test[0]
    X_train = df_train.drop(0, axis=1)
    X_test = df_test.drop(0, axis=1)

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'l1'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    print('Starting training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=20,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=5)

    print('Saving model...')
    # save model to file
    gbm.save_model('model.txt')

    print('Starting predicting...')
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # eval
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
    pass


def test_sklearn_example():
    print('Loading data...')
    # load or create your dataset
    df_train = pd.read_csv(r'/Users/longguangbin/Work/Codes/MLlearn/src/reg_models/LightGBM/data/regression.train',
                           header=None, sep='\t')
    df_test = pd.read_csv(r'/Users/longguangbin/Work/Codes/MLlearn/src/reg_models/LightGBM/data/regression.test',
                          header=None, sep='\t')

    y_train = df_train[0]
    y_test = df_test[0]
    X_train = df_train.drop(0, axis=1)
    X_test = df_test.drop(0, axis=1)

    print('Starting training...')
    # train
    gbm = lgb.LGBMRegressor(num_leaves=31,
                            learning_rate=0.05,
                            n_estimators=20)
    gbm.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='l1',
            early_stopping_rounds=5)

    print('Starting predicting...')
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
    # eval
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

    # feature importances
    print('Feature importances:', list(gbm.feature_importances_))

    # self-defined eval metric
    # f(y_true: array, y_pred: array) -> name: string, eval_result: float, is_higher_better: bool
    # Root Mean Squared Logarithmic Error (RMSLE)
    def rmsle(y_true, y_pred):
        return 'RMSLE', np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2))), False

    print('Starting training with custom eval function...')
    # train
    gbm.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=rmsle,
            early_stopping_rounds=5)

    print('Starting predicting...')
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
    # eval
    print('The rmsle of prediction is:', rmsle(y_test, y_pred)[1])

    # other scikit-learn modules
    estimator = lgb.LGBMRegressor(num_leaves=31)

    param_grid = {
        'learning_rate': [0.01, 0.1, 1],
        'n_estimators': [20, 40]
    }

    gbm = GridSearchCV(estimator, param_grid, cv=3)
    gbm.fit(X_train, y_train)

    print('Best parameters found by grid search are:', gbm.best_params_)


# no
def test_advanced_example():
    print('Loading data...')
    # load or create your dataset
    df_train = pd.read_csv(
        r'/Users/longguangbin/Work/Codes/MLlearn/src/reg_models/LightGBM/data/binary_classification/binary.train',
        header=None, sep='\t')
    df_test = pd.read_csv(
        r'/Users/longguangbin/Work/Codes/MLlearn/src/reg_models/LightGBM/data/binary_classification/binary.test',
        header=None, sep='\t')
    W_train = pd.read_csv(
        r'/Users/longguangbin/Work/Codes/MLlearn/src/reg_models/LightGBM/data/binary_classification/binary.train.weight',
        header=None)[0]
    W_test = pd.read_csv(
        r'/Users/longguangbin/Work/Codes/MLlearn/src/reg_models/LightGBM/data/binary_classification/binary.test.weight',
        header=None)[0]

    y_train = df_train[0]
    y_test = df_test[0]
    X_train = df_train.drop(0, axis=1)
    X_test = df_test.drop(0, axis=1)

    num_train, num_feature = X_train.shape

    # create dataset for lightgbm
    # if you want to re-use data, remember to set free_raw_data=False
    lgb_train = lgb.Dataset(X_train, y_train,
                            weight=W_train, free_raw_data=False)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train,
                           weight=W_test, free_raw_data=False)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    # generate feature names
    feature_name = ['feature_' + str(col) for col in range(num_feature)]

    print('Starting training...')
    # feature_name and categorical_feature
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10,
                    valid_sets=lgb_train,  # eval training data
                    feature_name=feature_name,
                    categorical_feature=[21])

    print('Finished first 10 rounds...')
    # check feature name
    print('7th feature name is:', lgb_train.feature_name[6])

    print('Saving model...')
    # save model to file
    gbm.save_model('model.txt')

    print('Dumping model to JSON...')
    # dump model to JSON (and save to file)
    model_json = gbm.dump_model()

    with open('model.json', 'w+') as f:
        json.dump(model_json, f, indent=4)

    # feature names
    print('Feature names:', gbm.feature_name())

    # feature importances
    print('Feature importances:', list(gbm.feature_importance()))

    print('Loading model to predict...')
    # load model to predict
    bst = lgb.Booster(model_file='model.txt')
    # can only predict with the best iteration (or the saving iteration)
    y_pred = bst.predict(X_test)
    # eval with loaded model
    print("The rmse of loaded model's prediction is:", mean_squared_error(y_test, y_pred) ** 0.5)

    print('Dumping and loading model with pickle...')
    # dump model with pickle
    with open('model.pkl', 'wb') as fout:
        pickle.dump(gbm, fout)
    # load model with pickle to predict
    with open('model.pkl', 'rb') as fin:
        pkl_bst = pickle.load(fin)
    # can predict with any iteration when loaded in pickle way
    y_pred = pkl_bst.predict(X_test, num_iteration=7)
    # eval with loaded model
    print("The rmse of pickled model's prediction is:", mean_squared_error(y_test, y_pred) ** 0.5)

    # continue training
    # init_model accepts:
    # 1. model file name
    # 2. Booster()
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10,
                    init_model='model.txt',
                    valid_sets=lgb_eval)

    print('Finished 10 - 20 rounds with model file...')

    # decay learning rates
    # learning_rates accepts:
    # 1. list/tuple with length = num_boost_round
    # 2. function(curr_iter)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10,
                    init_model=gbm,
                    learning_rates=lambda iter: 0.05 * (0.99 ** iter),
                    valid_sets=lgb_eval)

    print('Finished 20 - 30 rounds with decay learning rates...')

    # change other parameters during training
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10,
                    init_model=gbm,
                    valid_sets=lgb_eval,
                    callbacks=[lgb.reset_parameter(bagging_fraction=[0.7] * 5 + [0.6] * 5)])

    print('Finished 30 - 40 rounds with changing bagging_fraction...')

    # self-defined objective function
    # f(preds: array, train_data: Dataset) -> grad: array, hess: array
    # log likelihood loss
    def loglikelihood(preds, train_data):
        labels = train_data.get_label()
        preds = 1. / (1. + np.exp(-preds))
        grad = preds - labels
        hess = preds * (1. - preds)
        return grad, hess

    # self-defined eval metric
    # f(preds: array, train_data: Dataset) -> name: string, eval_result: float, is_higher_better: bool
    # binary error
    def binary_error(preds, train_data):
        labels = train_data.get_label()
        return 'error', np.mean(labels != (preds > 0.5)), False

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10,
                    init_model=gbm,
                    fobj=loglikelihood,
                    feval=binary_error,
                    valid_sets=lgb_eval)

    print('Finished 40 - 50 rounds with self-defined objective function and eval metric...')

    print('Starting a new training job...')

    # callback
    def reset_metrics():
        def callback(env):
            lgb_eval_new = lgb.Dataset(X_test, y_test, reference=lgb_train)
            if env.iteration - env.begin_iteration == 5:
                print('Add a new valid dataset at iteration 5...')
                env.model.add_valid(lgb_eval_new, 'new_valid')

        callback.before_iteration = True
        callback.order = 0
        return callback

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10,
                    valid_sets=lgb_train,
                    callbacks=[reset_metrics()])

    print('Finished first 10 rounds with callback function...')


# no
def test_logistic_regression():
    #################
    # Simulate some binary data with a single categorical and
    #   single continuous predictor
    np.random.seed(0)
    N = 1000
    X = pd.DataFrame({
        'continuous': range(N),
        'categorical': np.repeat([0, 1, 2, 3, 4], N / 5)
    })
    CATEGORICAL_EFFECTS = [-1, -1, -2, -2, 2]
    LINEAR_TERM = np.array([
        -0.5 + 0.01 * X['continuous'][k]
        + CATEGORICAL_EFFECTS[X['categorical'][k]] for k in range(X.shape[0])
    ]) + np.random.normal(0, 1, X.shape[0])
    TRUE_PROB = expit(LINEAR_TERM)
    Y = np.random.binomial(1, TRUE_PROB, size=N)
    DATA = {
        'X': X,
        'probability_labels': TRUE_PROB,
        'binary_labels': Y,
        'lgb_with_binary_labels': lgb.Dataset(X, Y),
        'lgb_with_probability_labels': lgb.Dataset(X, TRUE_PROB),
    }

    #################
    # Set up a couple of utilities for our experiments
    def log_loss(preds, labels):
        """Logarithmic loss with non-necessarily-binary labels."""
        log_likelihood = np.sum(labels * np.log(preds)) / len(preds)
        return -log_likelihood

    def experiment(objective, label_type, data):
        """Measure performance of an objective.
        Parameters
        ----------
        objective : string 'binary' or 'xentropy'
            Objective function.
        label_type : string 'binary' or 'probability'
            Type of the label.
        data : dict
            Data for training.
        Returns
        -------
        result : dict
            Experiment summary stats.
        """
        np.random.seed(0)
        nrounds = 5
        lgb_data = data['lgb_with_' + label_type + '_labels']
        params = {
            'objective': objective,
            'feature_fraction': 1,
            'bagging_fraction': 1,
            'verbose': -1
        }
        time_zero = time.time()
        gbm = lgb.train(params, lgb_data, num_boost_round=nrounds)
        y_fitted = gbm.predict(data['X'])
        y_true = data[label_type + '_labels']
        duration = time.time() - time_zero
        return {
            'time': duration,
            'correlation': np.corrcoef(y_fitted, y_true)[0, 1],
            'logloss': log_loss(y_fitted, y_true)
        }

    #################
    # Observe the behavior of `binary` and `xentropy` objectives
    print('Performance of `binary` objective with binary labels:')
    print(experiment('binary', label_type='binary', data=DATA))

    print('Performance of `xentropy` objective with binary labels:')
    print(experiment('xentropy', label_type='binary', data=DATA))

    print('Performance of `xentropy` objective with probability labels:')
    print(experiment('xentropy', label_type='probability', data=DATA))

    # Trying this throws an error on non-binary values of y:
    #   experiment('binary', label_type='probability', DATA)

    # The speed of `binary` is not drastically different than
    #   `xentropy`. `xentropy` runs faster than `binary` in many cases, although
    #   there are reasons to suspect that `binary` should run faster when the
    #   label is an integer instead of a float
    K = 10
    A = [experiment('binary', label_type='binary', data=DATA)['time']
         for k in range(K)]
    B = [experiment('xentropy', label_type='binary', data=DATA)['time']
         for k in range(K)]
    print('Best `binary` time: ' + str(min(A)))
    print('Best `xentropy` time: ' + str(min(B)))


# no
def test_plot_example():
    print('Loading data...')
    # load or create your dataset
    df_train = pd.read_csv(r'/Users/longguangbin/Work/Codes/MLlearn/src/reg_models/LightGBM/data/regression.train',
                           header=None, sep='\t')
    df_test = pd.read_csv(r'/Users/longguangbin/Work/Codes/MLlearn/src/reg_models/LightGBM/data/regression.test',
                          header=None, sep='\t')

    y_train = df_train[0]
    y_test = df_test[0]
    X_train = df_train.drop(0, axis=1)
    X_test = df_test.drop(0, axis=1)

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        'num_leaves': 5,
        'metric': ('l1', 'l2'),
        'verbose': 0
    }

    evals_result = {}  # to record eval results for plotting

    print('Starting training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=100,
                    valid_sets=[lgb_train, lgb_test],
                    feature_name=['f' + str(i + 1) for i in range(X_train.shape[-1])],
                    categorical_feature=[21],
                    evals_result=evals_result,
                    verbose_eval=10)

    print('Plotting metrics recorded during training...')
    ax = lgb.plot_metric(evals_result, metric='l1')
    plt.show()

    print('Plotting feature importances...')
    ax = lgb.plot_importance(gbm, max_num_features=10)
    plt.show()

    print('Plotting 84th tree...')  # one tree use categorical feature to split
    ax = lgb.plot_tree(gbm, tree_index=83, figsize=(20, 8), show_info=['split_gain'])
    plt.show()

    print('Plotting 84th tree with graphviz...')
    graph = lgb.create_tree_digraph(gbm, tree_index=83, name='Tree84')
    graph.render(view=True)
