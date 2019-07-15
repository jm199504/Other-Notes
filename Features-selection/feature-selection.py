# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:35:53 2019

@author: Junming Guo

Email: 2017223045154@stu.scu.edu.cn

Location: Chengdu, 610065 Sichuan Province, P. R. China
"""
# 过滤低方差特征
# 其中第一列特征被删除的原因在于第一列分别：0 0 1 0 0 0 
# 均值为1/6 经过计算方差为.138 低于阈值.16则删除

from sklearn.feature_selection import VarianceThreshold
X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
y = [0,0,0,1,1,1]
#print(X)
vt = VarianceThreshold(threshold=.16)
#print(vt.fit_transform(X))


# 单变量特征选择(考虑单个特征对模型效果选择k个特征)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
skb = SelectKBest(chi2, k=2).fit_transform(X, y)
#print(skb)

# 递归特征淘汰（结合预测模型）
# 其中返回rfe.ranking_表示各特征排名（1表示最高排名）
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X, y)
ranking = rfe.ranking_
#print(ranking)


# 使用模型特征选择（阈值设置越高选择特征数量越少）
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
svc = SVC(kernel="linear", C=1)
sfm = SelectFromModel(svc, threshold=1)
sfm.fit(X, y)
n_features = sfm.transform(X)
#print(n_features)

