# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 09:10:30 2019

@author: Junming Guo

Email: 2017223045154@stu.scu.edu.cn

Location: Chengdu, 610065 Sichuan Province, P. R. China
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# load data and split to train and test
iris = datasets.load_iris()
X,y = iris.data,iris.target
#alpha = 0.7
#train_length = int(alpha*(len(X)))
#train_X,test_X = X[:train_length],X[train_length:]
#test_mse,train_mse = list(),list()

# param dist
param_dist={"max_depth":[3,None],
            "criterion":['gini','entropy'],
            "min_samples_leaf":np.arange(1,11),
            "n_estimators":np.arange(5,20)}

# Prediction Model
clf=RandomForestClassifier()
n_iter_search = 300


# RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV
# param_dist字典类型，放入参数搜索范围
# n_iter=300，训练300次，数值越大，参数精度越大，搜索时间越长
# n_jobs = -1，使用所有的CPU进行训练，默认为1，使用1个CPU
random_search=RandomizedSearchCV(clf,param_distributions=param_dist,n_iter=n_iter_search)
random_search.fit(X,y)
print(random_search.best_params_)


# GridSearchCV

from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(estimator=clf, param_grid=param_dist, cv=5)
gs.fit(X,y)
print(gs.best_params_)


# BayesianOptimization | 贝叶斯超参优化
# 安装：pip install bayesian-optimization
# 出处：https://github.com/fmfn/BayesianOptimization

from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization

def rfc_cv(n_estimators, min_samples_split, max_features, data, targets):
    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=2
    )
    cval = cross_val_score(estimator, data, targets,
                           scoring='neg_log_loss', cv=4)
    return cval.mean()


def optimize_rfc(data, targets):
    def rfc_crossval(n_estimators, min_samples_split, max_features):
        return rfc_cv(
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=max(min(max_features, 0.999), 1e-3),
            data=data,
            targets=targets,
        )

    optimizer = BayesianOptimization(
        f=rfc_crossval,
        pbounds={
            "n_estimators": (10, 250),
            "min_samples_split": (2, 25),
            "max_features": (0.1, 0.999),
        },
        random_state=1234,
        verbose=2
    )
    optimizer.maximize(n_iter=10)

    print("Final result:", optimizer.max)

if __name__ == "__main__":
    iris = datasets.load_iris()
    X,y = iris.data,iris.target
    optimize_rfc(X, y)