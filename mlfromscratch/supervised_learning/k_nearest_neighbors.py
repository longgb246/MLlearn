# -*- coding:utf-8 -*-
from __future__ import print_function, division
import numpy as np
from mlfromscratch.utils import euclidean_distance


class KNN(object):
    """ K Nearest Neighbors classifier.

    Parameters:
    -----------
    k: int
        The number of closest neighbors that will determine the class of the 
        sample that we wish to predict.
    """

    def __init__(self, k=5):
        self.k = k

    def _vote(self, neighbor_labels):
        """ Return the most common class among the neighbor samples """
        counts = np.bincount(neighbor_labels.astype('int'))
        return counts.argmax()

    def predict(self, X_test, X_train, y_train):
        y_pred = np.empty(X_test.shape[0])
        # Determine the class of each sample
        for i, test_sample in enumerate(X_test):
            # Sort the training samples by their distance to the test sample and get the K nearest
            idx = np.argsort([euclidean_distance(test_sample, x) for x in X_train])[:self.k]
            # Extract the labels of the K nearest neighboring training samples
            k_nearest_neighbors = np.array([y_train[i] for i in idx])
            # Label sample as the most common class label
            y_pred[i] = self._vote(k_nearest_neighbors)

        return y_pred


def test_knn_reg():
    from sklearn.neighbors import KNeighborsRegressor

    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]

    neigh = KNeighborsRegressor(n_neighbors=2)
    neigh.fit(X, y)

    print(neigh.predict([[1.5]]))

    # 在k-NN回归中，k-NN算法用于估计连续变量。一种这样的算法使用k个最近邻居的加权平均值，通过它们的距离的倒数加权。
    # k个最近邻的加权平均值，权值为距离的倒数。
