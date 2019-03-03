# -*- coding:utf-8 -*-

from __future__ import print_function
import os
import sys
import os.path
import warnings

warnings.filterwarnings('ignore')

import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow

# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# from sklearn.metrics import accuracy_score
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, \
#     RobustScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import OneHotEncoder

# from xgboost import XGBClassifier

# from keras.preprocessing.sequence import TimeseriesGenerator
# from keras import optimizers
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization, LSTM, Embedding
# from keras.preprocessing.image import ImageDataGenerator
# from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
# from keras.utils import to_categorical

# from itertools import chain
# import imageio
# import skimage
# import skimage.io
# import skimage.transform
# from skimage.transform import rescale, resize, downscale_local_mean

path = '/Users/longguangbin/Work/Codes/MLlearn/kaggle/two_sigma_news_predict_stock/marketnewslstm-master/marketnewslstm'
os.chdir(path)
sys.path.append(path)

from MarketPrepro import MarketPrepro
from NewsPrepro import NewsPrepro
from JoinedGenerator import JoinedGenerator
from JoinedPreprocessor import JoinedPreprocessor
from ModelFactory import ModelFactory
from TrainValTestSplit import TrainValTestSplit
from Predictor import Predictor

# --------------- 相应设置 ---------------
plt.style.use('seaborn')
# Set random seed to make results reproducable
np.random.seed(42)
tensorflow.set_random_seed(42)
os.environ['PYTHONHASHSEED'] = '42'
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 200)


# --------------- 读取数据 ---------------
def read_data():
    input_dir = '../input'
    print(os.listdir("../input"))
    market = pd.read_csv(input_dir + '/marketdata_sample.csv')
    news = pd.read_csv(input_dir + '/news_sample.csv')
    # EDA
    print(market.columns)
    print(news.columns)
    # 设置
    news.time = pd.to_datetime('2007-02-01 23:35')
    # Restrict datetime to date
    # news.time = pd.to_datetime(news.time.astype('datetime64').dt.date, utc=True)
    news.time = news.time.astype('datetime64').dt.date
    market.time = market.time.astype('datetime64').dt.date
    return market, news


def split_data(market, toy):
    # Split to train, validation and test
    if toy:
        sample_size = 10000
    else:
        sample_size = 500000
    train_idx, val_idx, test_idx = TrainValTestSplit.train_val_test_split(market, sample_size)
    return train_idx, val_idx, test_idx


def main():
    toy = True

    market, news = read_data()
    train_idx, val_idx, test_idx = split_data(market, toy)

    # Create preprocessors
    market_prepro = MarketPrepro()
    market_prepro.fit(train_idx, market)
    news_prepro = NewsPrepro()
    news_prepro.fit(train_idx, news)
    prepro = JoinedPreprocessor(market_prepro, news_prepro)

    # Train data generator instance
    join_generator = JoinedGenerator(prepro, train_idx, market, news)
    val_generator = JoinedGenerator(prepro, val_idx, market, news)
    print('Generators created')

    # Create and train model
    model = ModelFactory.lstm_128(len(market_prepro.feature_cols) + len(news_prepro.feature_cols))
    model.load_weights("best_weights.h5")
    print(model.summary())
    ModelFactory.train(model, toy, join_generator, val_generator)

    # Predict
    predictor = Predictor(prepro, market_prepro, news_prepro, model, ModelFactory.look_back,
                          ModelFactory.look_back_step)
    y_pred, y_test = predictor.predict_idx(test_idx, market, news)

    y_pred = predictor.predict(market, news)

    plt.plot(y_pred)
    plt.plot(y_test)
    plt.legend(["pred", "test"])
    plt.show()

    # get_merged_Xy(train_idx.sample(5), market, pd.DataFrame([],columns=news.columns)).head()
    print('The end')


if __name__ == '__main__':
    main()
