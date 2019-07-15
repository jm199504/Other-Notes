## 数据归一化

### 1.最大最小归一化（min-max
normalization）

将数据映射到[0,1]或[-1,1]区间内，将有量纲的表达式转换成无量纲的表达式，即标量。

定义：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Normalization/images/1.png" width="300">

其中：min(x)：样本最小值，max(x)样本最大值，但是最大最小值容易受到异常点的影响，健壮性比较差，使用于传统的精确小数据。

### 2.均值归一化（mean normalization）

定义：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Normalization/images/2.png" width="300">

### 3.z-score

定义：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Normalization/images/3.png" width="300">

