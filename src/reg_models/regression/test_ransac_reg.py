# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/2/28
"""  
Usage Of 'test_ransac_reg.py' : 
"""

'''
逻辑伪代码:

[ Given ]:
    data – a set of observations
    model – a model to explain observed data points
    n – minimum number of data points required to estimate model parameters
    k – maximum number of iterations allowed in the algorithm
    t – threshold value to determine data points that are fit well by model 
    d – number of close data points required to assert that a model fits well to data

[ Return ]:
    bestFit – model parameters which best fit the data (or nul if no good model is found)

iterations = 0
bestFit = nul
bestErr = something really large
while iterations < k {
    maybeInliers = n randomly selected values from data
    maybeModel = model parameters fitted to maybeInliers
    alsoInliers = empty set
    for every point in data not in maybeInliers {
        if point fits maybeModel with an error smaller than t
             add point to alsoInliers
    }
    if the number of elements in alsoInliers is > d {
        % this implies that we may have found a good model
        % now test how good it is
        betterModel = model parameters fitted to all points in maybeInliers and alsoInliers
        thisErr = a measure of how well betterModel fits these points
        if thisErr < bestErr {
            bestFit = betterModel
            bestErr = thisErr
        }
    }
    increment iterations
}
return bestFit
'''

import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.linear_model import Lasso, Ridge


def ransac_regression(LR, features, targets, alpha, intercept=False):
    lr = LR(alpha=alpha, fit_intercept=intercept, normalize=True)
    rr = linear_model.RANSACRegressor(lr)
    rr.fit(features, targets)
    r2 = rr.score(features, targets)
    targets_hat = np.array(rr.predict(features))
    coefs = np.array(rr.estimator_.coef_)
    return targets_hat, r2, coefs, rr.estimator_.intercept_, rr
