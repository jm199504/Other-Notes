## 评估函数

<img src="https://github.com/jm199504/Other-Notes/blob/master/Evaluation-Function/images/1.png" width="500">

### 1.分类问题

#### Accuracy（准确率）

Accuracy = (TP + TN) / ( ALL )

#### Precision（精确率、查准率）

适用于识别垃圾邮件分类问题，尽可能减少对正常邮件的干预。

Precision = TP / (TP + FP)

#### Recall （召回率、查全率）

适用于金融风控领域中预测涨跌问题，尽可能降低频繁买入操作（预测为TP-上涨）。

Recall = TP / (TP + FN)

#### 综合评价 (F-Score)

<img src="https://github.com/jm199504/Other-Notes/blob/master/Evaluation-Function/images/2.png" width="300">

其中β=1时表示Precision和Recall权重同等重要，简称为F1-Score，β越大，Recall权重越大，反之Precision权重越大。

其中F1也称为精确率和召回率的调和均值

<img src="https://github.com/jm199504/Other-Notes/blob/master/Evaluation-Function/images/3.png" width="200">

即：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Evaluation-Function/images/4.png" width="300">

#### ROC/AUC

查看绘制ROC曲线个人笔记

#### 对数损失(Log Loss)

<img src="https://github.com/jm199504/Other-Notes/blob/master/Evaluation-Function/images/5.png" width="300">

其中N：样本数；M：类别数；yij：第i个样本属于分类j时为为1，否则为0；pij：第i个样本被预测为第j类的概率

#### 对比损失(Contrastive Loss)

常用于孪生LSTM（siamese LSTM）中，定义：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Evaluation-Function/images/6.png" width="300">

其中d= ||a-b||2表示其a与b（两个样本特征）的欧氏距离，y表示两样本是否匹配的标签，y=1表示样本相似或者匹配，反之为不匹配，margin为设定阈值（通常设为1）。

### 2.回归问题

#### 均方绝对误差MAE(Mean absolute error)

<img src="https://github.com/jm199504/Other-Notes/blob/master/Evaluation-Function/images/7.png" width="300">

```javascript
mae=np.sum(np.absolute(y_pred-y_test))/len(y_test)
```

#### 均方误差MSE(Mean Squared Error)

<img src="https://github.com/jm199504/Other-Notes/blob/master/Evaluation-Function/images/8.png" width="300">

```javascript
mse = np.sum((y_pred-y_test)**2)/len(y_test)
```

#### 均方根误差RMSE(Root means squared error)

<img src="https://github.com/jm199504/Other-Notes/blob/master/Evaluation-Function/images/9.png" width="300">

```javascript
rmse = mse_test ** 0.5
```

#### MAPE(Mean Absolute Percentage Error)

<img src="https://github.com/jm199504/Other-Notes/blob/master/Evaluation-Function/images/10.png" width="300">

```javascript
mape = 100 * np.sum(abs((y_pred-y_test)/y_test))/len(y_test)
```

#### 对称平均绝对百分比误差SMAPE(Symmetric Mean Absolute Percentage Error)

<img src="https://github.com/jm199504/Other-Notes/blob/master/Evaluation-Function/images/11.png" width="300">

```javascript
smape = 100 * np.sum(abs(y_pred-y_test)/(abs(y_test)+abs(y_pred)/2))/len(y_test)
```

#### R Squared(Coefficient of determination)

R Squared = 1 - MSE(y^,y) / Var(y)

```
rsquared = 1- mean_squared_error(y_test,y_preditc)/ np.var(y_test)
```

### 3.python代码实现

```
# classification problems

# Accuracy
Accuracy = (TP + TN) / ( TP + TN + FP + FN )

# Precision
Precision = TP / (TP + FP)

# Recall
Recall = TP / (TP + FN)

# F-score
beta = 1
fscore = (1+beta*beta)*precision*recall/((beta*beta)*precision+recall)

# regression problems

from sklearn import metrics

# MAPE
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100
    
# SMAPE
def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100
    
# MSE
print(metrics.mean_squared_error(y_true, y_pred)) 

# RMSE
print(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))

# MAE
print(metrics.mean_absolute_error(y_true, y_pred))

# MAPE
print(mape(y_true, y_pred))

# SMAPE
print(smape(y_true, y_pred))
```
