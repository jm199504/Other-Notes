# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# 获取数据
# 数据一
iris = datasets.load_iris()
X,y = iris.data,iris.target
# 数据二
#from sklearn.datasets import load_digits
#X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# 构建预测模型
nb = GaussianNB()
clf = RandomForestClassifier()
nb.fit(X_train, y_train)
clf.fit(X_train,y_train)

# 预测结果
predicted_probas = nb.predict_proba(X_test)
predict_result = clf.predict(X_test)
predicted_probas_rf = clf.predict_proba(X_test)

# 绘制ROC曲线图
import matplotlib.pyplot as plt
import scikitplot as skplt

#skplt.metrics.plot_roc(y_test, predicted_probas)
skplt.metrics.plot_roc_curve(y_test, predicted_probas_rf)

plt.show()