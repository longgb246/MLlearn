# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/19
"""  
Usage Of 'base_learn.py' : 
"""

from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

from collections import Counter
from terminaltables import AsciiTable

import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.examples.tutorials.mnist import input_data

print(tf.__version__)


def font(lg='ch', **kwargs):
    """ 使用中文 : lg='ch' """
    if lg == 'ch':
        font_v = FontProperties(fname='/Library/Fonts/STHeiti Light.ttc', **kwargs)
    else:
        font_v = FontProperties(**kwargs)
    return font_v


# -------------------------------------------------------------------------------
def base_1():
    # 构造器的返回值代表该常量 op 的返回值.
    matrix1 = tf.constant([[3., 3.]])

    # 创建另外一个常量 op, 产生一个 2x1 矩阵.
    matrix2 = tf.constant([[2.], [2.]])

    # 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入. 返回值 'product' 代表矩阵乘法的结果.
    product = tf.matmul(matrix1, matrix2)

    # 启动默认图.
    sess = tf.Session()

    # 返回值 'result' 是一个 numpy `ndarray` 对象.
    result = sess.run(product)
    print(result)

    # 任务完成, 关闭会话.
    sess.close()

    with tf.Session() as sess:
        result = sess.run([product])
        print(result)

    # 交互式使用
    sess = tf.InteractiveSession()

    x = tf.Variable([1.0, 2.0])
    a = tf.constant([3.0, 3.0])

    # 使用初始化器 initializer op 的 run() 方法初始化 'x'
    x.initializer.run()


def get_mnist():
    # 读取数据集，第一次TensorFlow会自动下载数据集到下面的路径中, label 采用 one_hot 形式
    mnist = input_data.read_data_sets('/Users/longguangbin/datasets/mnist', one_hot=True)

    # 1) 获得数据集的个数
    train_nums = mnist.train.num_examples
    test_nums = mnist.test.num_examples
    validation_nums = mnist.validation.num_examples
    print('MNIST数据集的个数')
    print('train_nums :', train_nums)
    print('test_nums :', test_nums)
    print('validation_nums :', validation_nums)

    # 2) 获得数据值
    train_data = mnist.train.images  # 所有训练数据
    val_data = mnist.validation.images  # (5000,784)
    test_data = mnist.test.images  # (10000,784)
    print('训练集数据大小：', train_data.shape)
    print('一副图像的大小：', train_data[0].shape)

    # 3) 获取标签值label=[0,0,...,0,1],是一个1*10的向量
    train_labels = mnist.train.labels  # (55000,10)
    val_labels = mnist.validation.labels  # (5000,10)
    test_labels = mnist.test.labels  # (10000,10)
    print('训练集标签数组大小：', train_labels.shape)
    print('一副图像的标签大小：', train_labels[1].shape)
    print('一副图像的标签值：', train_labels[0])

    # 4) 批量获取数据和标签【使用next_batch(batch_size)】
    batch_size = 100  # 每次批量训练100幅图像
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    print('使用mnist.train.next_batch(batch_size)批量读取样本')
    print('批量读取100个样本:数据集大小=', batch_xs.shape)
    print('批量读取100个样本:标签集大小=', batch_ys.shape)
    # xs是图像数据(100,784);ys是标签(100,10)

    # 5) 显示图像
    plt.figure()
    for i in range(100):
        im = train_data[i].reshape(28, 28)
        im = batch_xs[i].reshape(28, 28)
        plt.imshow(im, 'gray')
        plt.pause(0.0000001)
    plt.show()


# -------------------------------------------------------------------------------

# 基本分类：https://tensorflow.google.cn/tutorials/keras/basic_classification
def base_A_2():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # 0	T 恤衫/上衣
    # 1	裤子
    # 2	套衫
    # 3	裙子
    # 4	外套
    # 5	凉鞋
    # 6	衬衫
    # 7	运动鞋
    # 8	包包
    # 9	踝靴

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # 基本的 summary
    print('train labels : \n\t', dict(Counter(train_labels)))
    print('test labels : \n\t', dict(Counter(test_labels)))
    print('train images shape : ', train_images.shape)
    print('train_labels images shape : ', train_labels.shape)
    print('test images shape : ', test_images.shape)
    print('test_labels images shape : ', test_labels.shape)

    # 数据归一化
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # 画其中一幅图
    def s_plot_one_figure():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        img1 = ax.imshow(train_images[0], cmap='Blues')
        fig.colorbar(img1, ax=ax)  # img + ax -> fig 添加 colorbar
        ax.grid(False)
        plt.show()

    s_plot_one_figure()

    # 画前32幅图
    def s_plot_32_figures():
        fig = plt.figure(figsize=(15, 8))
        for i in range(32):
            ax = fig.add_subplot(4, 8, i + 1)
            img1 = ax.imshow(train_images[i], cmap='Blues')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_xlabel(class_names[train_labels[i]])
        fig.suptitle('32 Figures Of Clothes', y=0.95, fontsize=15)
        # fig.suptitle(u'32 幅服饰的图片', y=0.95, fontproperties=FontProperties(
        #     fname='/Library/Fonts/STHeiti Light.ttc', size=15))
        # fig.suptitle(u'32 幅服饰的图片', y=0.95, fontproperties=font(size=15))

    s_plot_32_figures()

    # 设置模型 : 最终要出 10 个分类，所以输出的节点数为 10 个
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),  # 扁平化数据
        keras.layers.Dense(128, activation=tf.nn.relu),  # 128 个节点（或神经元），隐藏层
        keras.layers.Dense(10, activation=tf.nn.softmax)  # 10 个节点（或神经元）
    ])

    # 编译模型
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 训练模型 : TODO - 猜测是不是自动将 labels 进行了转化，计算一个数字与一个向量的 loss，思考一下。
    model.fit(train_images, train_labels, epochs=5)

    # 测试数据的模型表现
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

    # 预测
    predictions = model.predict(test_images)
    print('Predictions.shape : ', predictions.shape)
    print('First predict : ', np.argmax(predictions[0]))  # 结果 : 预测值最大的 TODO - 理清楚原因
    print('First label : ', test_labels[0])

    # 画图看结果
    def plot_image(i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],  # 预测的label
                                             100 * np.max(predictions_array),  # 预测为该label的概率
                                             class_names[true_label]), color=color)  # 实际的label

    def plot_value_array(i, predictions_array, true_label):
        predictions_array, true_label = predictions_array[i], true_label[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")  # 画出每个类别的概率
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')  # 预测的标签是red
        thisplot[true_label].set_color('blue')  # 正确的标签是blue

    def plot_one_acc(i, predictions, test_labels, test_images):
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(1, 2, 2)
        plot_value_array(i, predictions, test_labels)

    def plot_multi_acc(row, col, predictions, test_labels, test_images, s=0):
        num_rows = row
        num_cols = col
        num_images = num_rows * num_cols
        plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        for i in range(s, num_images + s):
            plt.subplot(num_rows, 2 * num_cols, 2 * (i - s) + 1)
            plot_image(i, predictions, test_labels, test_images)
            plt.subplot(num_rows, 2 * num_cols, 2 * (i - s) + 2)
            plot_value_array(i, predictions, test_labels)

    plot_one_acc(12, predictions, test_labels, test_images)
    plot_multi_acc(5, 3, predictions, test_labels, test_images)

    # 预测单个
    def predict_one():
        img = test_images[0]
        print(img.shape)
        img = (np.expand_dims(img, 0))
        print(img.shape)

        predictions_single = model.predict(img)
        print(predictions_single)

        plot_one_acc(0, predictions_single, test_labels, img)

    predict_one()


# 基本文本分类：https://tensorflow.google.cn/tutorials/keras/basic_text_classification
def base_A_3():
    # IMDB 数据集 : 来自互联网电影数据库的 50000 条影评文本。我们将这些影评拆分为训练集（25000 条影评）和测试集（25000 条影评）
    imdb = keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    pass


# 回归：https://tensorflow.google.cn/tutorials/keras/basic_regression
def base_A_4():
    # Auto MPG 数据集
    dataset_path = keras.utils.get_file("auto-mpg.data",
                                        "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    print(dataset_path)  # '/Users/longguangbin/.keras/datasets/auto-mpg.data'
    dataset_path = '/Users/longguangbin/.keras/datasets/auto-mpg.data'

    # 读取 Auto MPG 数据集
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']
    raw_dataset = pd.read_csv(dataset_path, names=column_names,
                              na_values="?", comment='\t',
                              sep=" ", skipinitialspace=True)
    dataset = raw_dataset.copy()
    dataset.tail()

    # EDA
    dataset.isna().sum()  # 检查为 null 的个数
    dataset = dataset.dropna()  # 去掉 null 的行数
    dataset.index = range(len(dataset))
    print('dataset columns : ', dataset.columns)

    # 变量转成 one-hot
    origin = dataset.pop('Origin')
    dataset['USA'] = (origin == 1) * 1.0
    dataset['Europe'] = (origin == 2) * 1.0
    dataset['Japan'] = (origin == 3) * 1.0
    dataset.tail()

    # 拆分成 train 和 test
    train_dataset = dataset.sample(frac=0.8, random_state=0)  # 随机抽样
    test_dataset = dataset.drop(train_dataset.index)

    # 检查数据
    sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
    train_stats = train_dataset.describe()
    train_stats.pop("MPG")
    train_stats = train_stats.transpose()
    pd.set_option('display.max_columns', 40)  # 设置打印宽度
    print(train_stats)

    # 取出 label 目标变量 - MPG
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')

    def norm(x, train_stats):
        return (x - train_stats['mean']) / train_stats['std']

    # 标准化数据
    normed_train_data = norm(train_dataset, train_stats)
    normed_test_data = norm(test_dataset, train_stats)

    # 简历模型
    def build_model():
        model = keras.Sequential([
            layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
            layers.Dense(64, activation=tf.nn.relu),
            layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.RMSprop(0.001)

        model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=['mean_absolute_error', 'mean_squared_error'])
        return model

    model = build_model()
    model.summary()  # 查看参数等个数

    example_batch = normed_train_data[:10]
    example_result = model.predict(example_batch)
    print(example_result)

    # 训练模型
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0:
                print('')
            print('.', end='')

    EPOCHS = 1000

    history = model.fit(normed_train_data, train_labels,
                        epochs=EPOCHS, validation_split=0.2, verbose=0,
                        callbacks=[PrintDot()])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    # 画出 error 的变化
    def plot_history(history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        fig = plt.figure(figsize=(14, 7))
        ax1 = fig.add_subplot(121)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Mean Abs Error [MPG]')
        ax1.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
        ax1.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Val Error')
        ax1.set_ylim([0, 5])
        ax1.set_title('Mean Abs Error [MPG]')
        ax1.legend()

        ax2 = fig.add_subplot(122)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Square Error [$MPG^2$]')
        ax2.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
        ax2.plot(hist['epoch'], hist['val_mean_squared_error'], label='Val Error')
        ax2.set_ylim([0, 20])
        ax2.set_title('Mean Square Error [$MPG^2$]')
        ax2.legend()

        plt.tight_layout(rect=(0.05, 0.05, 0.95, 0.95))
        plt.show()

    plot_history(history)
    # 图表显示在约100个时期之后，验证错误几乎没有改善，甚至降低

    model = build_model()

    # 提前终止，当验证的loss - val_loss 不再增加的时候
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    # The patience parameter is the amount of epochs to check for improvement

    history = model.fit(normed_train_data, train_labels,
                        epochs=EPOCHS, validation_split=0.2, verbose=0,
                        callbacks=[early_stop, PrintDot()])

    plot_history(history)

    # 评估
    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
    print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

    # 做预测
    def plot_predictions(test_labels, test_predictions):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(test_labels, test_predictions)
        ax.set_xlabel('True Values [MPG]')
        ax.set_ylabel('Predictions [MPG]')
        ax.axis('equal')
        ax.axis('square')
        ax.set_xlim([0, ax.get_xlim()[1]])
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.plot([-100, 100], [-100, 100])

    def plot_error_hist(test_labels, test_predictions):
        error = test_predictions - test_labels
        plt.hist(error, bins=25)
        plt.xlabel("Prediction Error [MPG]")
        plt.ylabel("Count")

    test_predictions = model.predict(normed_test_data).flatten()

    plot_predictions(test_labels, test_predictions)
    plot_error_hist(test_labels, test_predictions)

    # 早期停止是防止过度拟合的有用技术。


# 过拟合、欠拟合：https://tensorflow.google.cn/tutorials/keras/overfit_and_underfit
def base_A_5():
    # 欠拟合: 1.模型不够强大，2.过于正则化，3.根本没有训练足够长的时间
    # 防止过拟合: 1.使用更多训练数据, 2.正则化, (权重正则化, 丢弃)

    NUM_WORDS = 10000

    (train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

    def multi_hot_sequences(sequences, dimension):
        # Create an all-zero matrix of shape (len(sequences), dimension)
        results = np.zeros((len(sequences), dimension))
        for i, word_indices in enumerate(sequences):
            results[i, word_indices] = 1.0  # set specific indices of results[i] to 1s
        return results

    train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
    test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)
    pass


# 保存模型：https://tensorflow.google.cn/tutorials/keras/save_and_restore_models
def base_A_6():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]

    train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
    test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

    # Returns a short sequential model
    def create_model():
        model = tf.keras.models.Sequential([
            keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy'])

        return model

    # Create a basic model instance
    model = create_model()
    model.summary()
    pass


def base_B_1():
    tf.enable_eager_execution()


def base_B_2():
    tf.enable_eager_execution()

# 1、搞定自营预测的流程
# 2、tf学习精通
# 3、算法 - nlp、推荐、cv等等
# 4、java
