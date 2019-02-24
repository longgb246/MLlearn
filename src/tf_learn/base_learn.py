# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/19
"""  
Usage Of 'base_learn.py' : 
"""

from __future__ import print_function, division

import tensorflow as tf
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data

print(tf.__version__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from collections import Counter
from terminaltables import AsciiTable


def font(lg='ch', **kwargs):
    """ 使用中文 : lg='ch' """
    if lg == 'ch':
        font_v = FontProperties(fname='/Library/Fonts/STHeiti Light.ttc', **kwargs)
    else:
        font_v = FontProperties(**kwargs)
    return font_v


# ------------------------------------------------------------
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


# https://tensorflow.google.cn/tutorials/keras/basic_classification
def base_2():
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
    fig = plt.figure()
    ax = fig.add_subplot(111)
    img1 = ax.imshow(train_images[0], cmap='Blues')
    fig.colorbar(img1, ax=ax)  # img + ax -> fig 添加 colorbar
    ax.grid(False)
    plt.show()

    # 画前32幅图
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

    # 设置模型
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # 编译模型
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 训练模型
    model.fit(train_images, train_labels, epochs=5)

    # 测试数据的模型表现
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)

    predictions = model.predict(test_images)
    predictions.shape


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
