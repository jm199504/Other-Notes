## 损失函数

### 1.均方误差/平方损失/L2损失（MSE）

<img src="https://github.com/jm199504/Other-Notes/blob/master/Loss-Function/images/1.png" width="200">

MSE度量预测值和时间观测值的平方差的均值，只考虑了误差的平均大小，而没有考虑方向。

代码实现：

```javascript
import numpy as np
y_hat = np.array([0.000, 0.166, 0.333])
y_true = np.array([0.000, 0.254, 0.998])
def rmse(predictions, targets):
    differences = predictions - targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)
    return rmse_val
print("d is: " + str(["%.8f" % elem for elem in y_hat]))
print("p is: " + str(["%.8f" % elem for elem in y_true]))
rmse_val = rmse(y_hat, y_true)
print("rms error is: " + str(rmse_val))
```

### 2.平均绝对误差/L1 损失

<img src="https://github.com/jm199504/Other-Notes/blob/master/Loss-Function/images/2.png" width="200">

平均绝对误差（MAE）度量的是预测值和实际观测值之间绝对差之和的平均值。和 MSE 一样，这种度量方法也是在不考虑方向的情况下衡量误差大小。但和 MSE 的不同之处在于，MAE 需要像线性规划这样更复杂的工具来计算梯度。此外，MAE 对异常值更加稳健，因为它不使用平方。

代码实现：

```javascript
import numpy as np
y_hat = np.array([0.000, 0.166, 0.333])
y_true = np.array([0.000, 0.254, 0.998])

print("d is: " + str(["%.8f" % elem for elem in y_hat]))
print("p is: " + str(["%.8f" % elem for elem in y_true]))

def mae(predictions, targets):
    differences = predictions - targets
    absolute_differences = np.absolute(differences)
    mean_absolute_differences = absolute_differences.mean()
    return mean_absolute_differences
mae_val = mae(y_hat, y_true)
print ("mae error is: " + str(mae_val))
```

### 3.平均偏差误差（mean bias error）

<img src="https://github.com/jm199504/Other-Notes/blob/master/Loss-Function/images/3.png" width="200">

与 MAE 相似，唯一的区别是这个函数没有用绝对值。用这个函数需要注意的一点是，正负误差可以互相抵消。尽管在实际应用中没那么准确，但它可以确定模型存在正偏差还是负偏差。

### 4.Hinge Loss/多分类 SVM 损失

<img src="https://github.com/jm199504/Other-Notes/blob/master/Loss-Function/images/4.png" width="200">

hinge loss 常用于最大间隔分类（maximum-margin classification）（最常用：支持向量机）。

尽管不可微，但它是一个凸函数，因此可以轻而易举地使用机器学习领域中常用的凸优化器。

计算过程举例：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Loss-Function/images/5.png">

其中预测3幅图的真实标签分别为：dog、cat、horse，其以下表格表示模型预测各类别的值，计算过程如下：

> #### 1st training example
>
> max(0, (1.49) - (-0.39) + 1) + max(0, (4.21) - (-0.39) + 1)
> max(0, 2.88) + max(0, 5.6)
> 2.88 + 5.6
> 8.48 #(High loss as very wrong prediction)
>
> #### 2nd training example
>
> max(0, (-4.61) - (3.28)+ 1) + max(0, (1.46) - (3.28)+ 1)
> max(0, -6.89) + max(0, -0.82)
> 0 + 0
> 0 #(Zero loss as correct prediction)
>
> #### 3rd training example
>
> max(0, (1.03) - (-2.27)+ 1) + max(0, (-2.37) - (-2.27)+ 1)
> max(0, 4.3) + max(0, 0.9)
> 4.3 + 0.9
> 5.2 #(High loss as very wrong prediction)

### 5.交叉熵损失/负对数似然

<img src="https://github.com/jm199504/Other-Notes/blob/master/Loss-Function/images/6.png" width="200">

分类问题中最常见的设置。随着预测概率偏离实际标签，交叉熵损失会逐渐增加。

当实际标签为 1(y(i)=1) 时，函数的后半部分消失，而当实际标签是为 0(y(i=0)) 时，函数的前半部分消失。简言之，我们只是把对真实值类别的实际预测概率的对数相乘。还有重要的一点是，交叉熵损失会重重惩罚那些置信度高但是错误的预测值。

代码实现：

```javascript
import numpy as np
predictions = np.array([[0.25,0.25,0.25,0.25],
                        [0.01,0.01,0.01,0.96]])
targets = np.array([[0,0,0,1],
                   [0,0,0,1]])
def cross_entropy(predictions, targets, epsilon=1e-10):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce_loss = -np.sum(np.sum(targets * np.log(predictions + 1e-5)))/N
    return ce_loss
cross_entropy_loss = cross_entropy(predictions, targets)
print ("Cross entropy loss is: " + str(cross_entropy_loss))
```
