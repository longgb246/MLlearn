# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : lgb453476610@163.com
  Date    : 2018/10/25
  Usage   :
"""

import json
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


def test_simple1():
    # 加载数据
    print('Load data...')

    iris = load_iris()
    data = iris.data
    target = iris.target
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

    # df_train = pd.read_csv('../regression/regression.train', header=None, sep='\t')
    # df_test = pd.read_csv('../regression/regression.test', header=None, sep='\t')
    # y_train = df_train[0].values
    # y_test = df_test[0].values
    # X_train = df_train.drop(0, axis=1).values
    # X_test = df_test.drop(0, axis=1).values

    print('Start training...')
    # 创建模型，训练模型
    gbm = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20)
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='l1', early_stopping_rounds=5)

    print('Start predicting...')
    # 测试机预测
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
    # 模型评估
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

    # feature importances
    print('Feature importances:', list(gbm.feature_importances_))

    # 网格搜索，参数优化
    estimator = lgb.LGBMRegressor(num_leaves=31)

    param_grid = {
        'learning_rate': [0.01, 0.1, 1],
        'n_estimators': [20, 40]
    }

    gbm = GridSearchCV(estimator, param_grid)

    gbm.fit(X_train, y_train)

    print('Best parameters found by grid search are:', gbm.best_params_)


def test_simple2():
    iris = load_iris()
    data = iris.data
    target = iris.target
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

    # 加载你的数据
    # print('Load data...')
    # df_train = pd.read_csv('../regression/regression.train', header=None, sep='\t')
    # df_test = pd.read_csv('../regression/regression.test', header=None, sep='\t')
    #
    # y_train = df_train[0].values
    # y_test = df_test[0].values
    # X_train = df_train.drop(0, axis=1).values
    # X_test = df_test.drop(0, axis=1).values

    # 创建成lgb特征的数据集格式
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # 将参数写成字典下形式
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'regression',  # 目标函数
        'metric': {'l2', 'auc'},  # 评估函数
        'num_leaves': 31,  # 叶子节点数
        'learning_rate': 0.05,  # 学习速率
        'feature_fraction': 0.9,  # 建树的特征选择比例
        'bagging_fraction': 0.8,  # 建树的样本采样比例
        'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }

    print('Start training...')
    # 训练 cv and train
    gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)
    # gbm = lgb.train(params, lgb_train, num_boost_round=20, early_stopping_rounds=5)

    print('Save model...')
    # 保存模型到文件
    gbm.save_model('model.txt')

    print('Start predicting...')
    # 预测数据集
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # 评估模型
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
