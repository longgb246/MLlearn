# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : lgb453476610@163.com
  Date    : 2018/10/25
  Usage   :
"""

import xgboost as xgb
from sklearn import datasets
from sklearn.cross_validation import train_test_split


def test_xgboost():
    iris = datasets.load_iris()
    data = iris.data[:100]
    print(data.shape)
    # (100L, 4L)
    # 一共有100个样本数据, 维度为4维
    label = iris.target[:100]
    print(label)

    train_x, test_x, train_y, test_y = train_test_split(data, label, random_state=0)

    # train_y2 = np.random.rand(len(train_y)) * 100

    # dtrain = xgb.DMatrix(train_x, label=train_y2)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x)

    params = {'booster': 'gbtree',
              # 'objective': 'binary:logistic',
              'objective': 'reg:gamma',
              'eval_metric': 'auc',
              'max_depth': 4,
              'lambda': 10,
              'subsample': 0.75,
              'colsample_bytree': 0.75,
              'min_child_weight': 2,
              'eta': 0.025,
              'seed': 0,
              'learning_rate': 0.1,
              'nthread': 8,
              'silent': 1}

    watchlist = [(dtrain, 'train')]

    # bst = xgb.train(params, dtrain, num_boost_round=100, evals=watchlist)
    bst = xgb.train(params, dtrain, num_boost_round=100)

    ypred = bst.predict(dtest)

    # 设置阈值, 输出一些评价指标
    y_pred = (ypred >= 0.5) * 1

    from sklearn import metrics

    print('AUC: %.4f' % metrics.roc_auc_score(test_y, ypred))
    print('ACC: %.4f' % metrics.accuracy_score(test_y, y_pred))
    print('Recall: %.4f' % metrics.recall_score(test_y, y_pred))
    print('F1-score: %.4f' % metrics.f1_score(test_y, y_pred))
    print('Precesion: %.4f' % metrics.precision_score(test_y, y_pred))
    metrics.confusion_matrix(test_y, y_pred)


def test_xgboost2():
    import pandas as pd
    import matplotlib.pyplot as plt
    import xgboost as xgb
    import numpy as np
    from xgboost import plot_importance
    from sklearn.preprocessing import Imputer


    def loadDataset(filePath):
        df = pd.read_csv(filepath_or_buffer=filePath)
        return df


    def featureSet(data):
        data_num = len(data)
        XList = []
        for row in range(0, data_num):
            tmp_list = []
            tmp_list.append(data.iloc[row]['club'])
            tmp_list.append(data.iloc[row]['league'])
            tmp_list.append(data.iloc[row]['potential'])
            tmp_list.append(data.iloc[row]['international_reputation'])
            XList.append(tmp_list)
        yList = data.y.values
        return XList, yList


    def loadTestData(filePath):
        data = pd.read_csv(filepath_or_buffer=filePath)
        data_num = len(data)
        XList = []
        for row in range(0, data_num):
            tmp_list = []
            tmp_list.append(data.iloc[row]['club'])
            tmp_list.append(data.iloc[row]['league'])
            tmp_list.append(data.iloc[row]['potential'])
            tmp_list.append(data.iloc[row]['international_reputation'])
            XList.append(tmp_list)
        return XList


    def trainandTest(X_train, y_train, X_test):
        # XGBoost训练过程
        model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False, objective='reg:gamma')
        model.fit(X_train, y_train)

        # 对测试集进行预测
        ans = model.predict(X_test)

        ans_len = len(ans)
        id_list = np.arange(10441, 17441)
        data_arr = []
        for row in range(0, ans_len):
            data_arr.append([int(id_list[row]), ans[row]])
        np_data = np.array(data_arr)

        # 写入文件
        pd_data = pd.DataFrame(np_data, columns=['id', 'y'])
        # print(pd_data)
        pd_data.to_csv('submit.csv', index=None)

        # 显示重要特征
        # plot_importance(model)
        # plt.show()

if __name__ == '__main__':
    trainFilePath = 'dataset/soccer/train.csv'
    testFilePath = 'dataset/soccer/test.csv'
    data = loadDataset(trainFilePath)
    X_train, y_train = featureSet(data)
    X_test = loadTestData(testFilePath)
    trainandTest(X_train, y_train, X_test)
