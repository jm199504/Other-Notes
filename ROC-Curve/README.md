## 绘制ROC曲线图

### 1.安装库

Scikit-plot：pip install scikit-plot

### 2.仓库地址

https://github.com/reiinakano/scikit-plot

### 3.提供示例

```javascript
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
nb = GaussianNB()
nb.fit(X_train, y_train)
predicted_probas = nb.predict_proba(X_test)
*# The magic happens here*
import matplotlib.pyplot as plt
import scikitplot as skplt
skplt.metrics.plot_roc(y_test, predicted_probas)
plt.show()
```

### 4.实际操作

```javascript
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
```

### 5.输出效果图

<img src="https://github.com/jm199504/Other-Notes/blob/master/ROC-Curve/images/1.png">

### 6.其他示范图

#### 6.1 P-R曲线

精确率precision vs 召回率recall 曲线，以recall作为横坐标轴，precision作为纵坐标轴。

```javascript
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits as load_data
import scikitplot as skplt
# Load dataset
X, y = load_data(return_X_y=True)
# Create classifier instance then fit
nb = GaussianNB()
nb.fit(X,y)
# Get predicted probabilities
y_probas = nb.predict_proba(X)
skplt.metrics.plot_precision_recall_curve(y, y_probas, cmap='nipy_spectral')
plt.show()
```

输出图：

<img src="https://github.com/jm199504/Other-Notes/blob/master/ROC-Curve/images/2.png" width="500">

#### 6.2 混淆矩阵

是分类的重要评价标准，下面代码是用随机森林对鸢尾花数据集进行分类，分类结果画一个归一化的混淆矩阵。

```javascript
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits as load_data
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import scikitplot as skplt
X, y = load_data(return_X_y=True)
# Create an instance of the RandomForestClassifier
classifier = RandomForestClassifier()
# Perform predictions
predictions = cross_val_predict(classifier, X, y)
plot = skplt.metrics.plot_confusion_matrix(y, predictions, normalize=True)
plt.show()
```

输出图：

<img src="https://github.com/jm199504/Other-Notes/blob/master/ROC-Curve/images/3.png" width="500">

### 7.补充
AUC值是ROC曲线（接收者操作特征曲线（receiver operating characteristic curve）下的面积，用来评判模型结果结果的可信度，可以理解为，在样本里抽一个正样本和一个负样本，正样本的评分高于负样本的概率比较大。

ROC曲线的横坐标是伪阳性率（也叫假正类率，False Positive Rate），纵坐标是真阳性率（真正类率，True Positive Rate）。

（1）伪阳性率（FPR），也称假阳性率(False Positice Rate，FPR)，误诊率( = 1 - 特异度)：

判定为正例却不是真正例的概率（负例被预测为正例）

<img src="https://github.com/jm199504/Interview-bible/blob/master/images/15.png">

（2）真阳性率（TPR），也称真阳性率(True Positive Rate，TPR)，灵敏度(Sensitivity)，召回率(Recall)：

判定为正例也是真正例的概率（正例被预测为正例）

<img src="https://github.com/jm199504/Interview-bible/blob/master/images/12.png">

基于预测结果对ROC可视化效果图：

<img src="https://github.com/jm199504/Other-Notes/blob/master/ROC-Curve/images/4.png">
