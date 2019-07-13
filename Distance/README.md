## 1.距离度量 & 相似性度量(distance & similarity measurement)

> 欧氏距离
>
> 曼哈顿距离
>
> 切比雪夫距离
>
> 闵可夫斯基距离
>
> 标准化欧氏距离
>
> 马氏距离
>
> 夹角余弦
>
> 汉明距离
>
> 杰卡德距离&杰卡德相似系数
>
> 相关系数& 相关距离

**1.1欧氏距离(EuclideanDistance)**

欧氏距离是最易于理解的一种距离计算方法，源自欧氏空间中两点间的距离公式。

(1)二维平面上两点a(x1,y1)与b(x2,y2)间的欧氏距离：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/1.png">

(2)三维空间两点a(x1,y1,z1)与b(x2,y2,z2)间的欧氏距离：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/2.png">

(3)两个n维向量a(x11,x12,…,x1n)与 b(x21,x22,…,x2n)间的欧氏距离：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/3.png">

也可以用表示成向量运算的形式：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/4.png">

**1.2曼哈顿距离(ManhattanDistance)**

想象你在曼哈顿要从一个十字路口开车到另外一个十字路口，驾驶距离是两点间的直线距离吗？显然不是，除非你能穿越大楼。实际驾驶距离是你走过马路距离，而曼哈顿的马路均为直线(十字路口)，就是这个“曼哈顿距离”。而这也是曼哈顿距离名称的来源， 曼哈顿距离也称为城市街区距离(CityBlock distance)。

举个栗子：点和橙点曼哈顿距离 = 穿过的横竖直线和

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/5.png" width="300">

(1)二维平面两点a(x1,y1)与b(x2,y2)间的曼哈顿距离

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/6.png">

(2)两个n维向量a(x11,x12,…,x1n)与b(x21,x22,…,x2n)间的曼哈顿距离

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/7.png">

**1.3切比雪夫距离 ( Chebyshev Distance )**

假设平面空间有2点，分别为(x1,y1)，(x2,y2)。两点的横纵坐标差的最大值max(| x2-x1 | , | y2-y1 | ) ，即切比雪夫距离。

(1)二维平面两点a(x1,y1)与b(x2,y2)间的切比雪夫距离

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/8.png">

(2)两个n维向量a(x11,x12,…,x1n)与b(x21,x22,…,x2n)间的切比雪夫距离

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/9.png">

另一种等价形式是

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/10.png">

**1.4闵可夫斯基距离(MinkowskiDistance)**

闵氏距离不是一种距离，而是一组距离的定义。

(1)闵氏距离的定义

两个n维变量a(x11,x12,…,x1n)与b(x21,x22,…,x2n)间的闵可夫斯基距离定义为：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/11.png">

其中p是一个变参数。

当p=1时，就是曼哈顿距离

当p=2时，就是欧氏距离

当p→∞时，就是切比雪夫距离

根据变参数的不同，闵氏距离可以表示一类的距离。

(2)闵氏距离的缺点

闵氏距离，包括曼哈顿距离、欧氏距离和切比雪夫距离都存在明显的缺点。

举个例子：二维样本(身高,体重)，其中身高范围是150~190，体重范围是50~60，有三个样本：a(180,50)，b(190,50)，c(180,60)。那么a与b之间的闵氏距离（无论是曼哈顿距离、欧氏距离或切比雪夫距离）等于a与c之间的闵氏距离，但是身高的10cm真的等价于体重的10kg么？因此用闵氏距离来衡量这些样本间的相似度很有问题。
闵氏距离的缺点主要有两个：

(1)将各个分量的量纲(scale)，也就是“单位”当作相同的看待了。

(2)没有考虑各个分量的分布（期望，方差等)可能是不同的。

**1.5标准化欧氏距离(Standardized Euclidean distance )**

(1)标准欧氏距离的定义

标准化欧氏距离是针对简单欧氏距离的缺点而作的一种改进方案。标准欧氏距离的思路：既然数据各维分量的分布不一样，好吧！那我先将各个分量都“标准化”到均值、方差相等吧。均值和方差标准化到多少呢？这里先复习点统计学知识吧，假设样本集X的均值(mean)为m，标准差(standarddeviation)为s，那么X的“标准化变量”表示为

标准化变量的数学期望为0，方差为1。因此样本集的标准化过程(standardization)用公式描述就是：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/12.png">

标准化后的值 =  ( 标准化前的值  － 分量的均值 ) /分量的标准差

经过简单的推导就可以得到两个n维向量a(x11,x12,…,x1n)与b(x21,x22,…,x2n)间的标准化欧氏距离的公式：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/13.png">

如果将方差的倒数看成是一个权重，这个公式可以看成是一种加权欧氏距离(WeightedEuclidean distance)。

**1.6马哈拉诺比斯距离(MahalanobisDistance,简称马氏距离)**

（1）马氏距离定义：有M个样本向量X1~Xm，协方差矩阵记为S，均值记为向量μ，则其中样本向量X到u的马氏距离表示为：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/14.png">

而其中向量Xi与Xj之间的马氏距离定义为：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/15.png">

若协方差矩阵是单位矩阵（各个样本向量之间独立同分布）,则公式就成了：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/16.png">

即欧氏距离

若协方差矩阵是对角矩阵，公式变成了标准化欧氏距离。

(2)马氏距离的优缺点：量纲无关，排除变量之间的相关性的干扰。

1.7夹角余弦(Cosine)

几何中夹角余弦可用来衡量两个向量方向的差异，机器学习中借用这一概念来衡量样本向量之间的差异。

(1)在二维空间中向量A(x1,y1)与向量B(x2,y2)的夹角余弦公式：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/17.png">

(2)两个n维样本点a(x11,x12,…,x1n)和b(x21,x22,…,x2n)的夹角余弦
类似的，对于两个n维样本点a(x11,x12,…,x1n)和b(x21,x22,…,x2n)，可以使用类似于夹角余弦的概念来衡量它们间的相似程度。

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/18.png">

即：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/19.png">

夹角余弦取值范围为[-1,1]。夹角余弦越大表示两个向量的夹角越小，夹角余弦越小表示两向量的夹角越大。当两个向量的方向重合时夹角余弦取最大值1，当两个向量的方向完全相反夹角余弦取最小值-1。

1.8汉明距离(Hammingdistance)

汉明距离的定义：两个等长字符串s1与s2之间的汉明距离定义为将其中一个变为另外一个所需要作的最小替换次数。例如字符串“1111”与“1001”之间的汉明距离为2。

应用：信息编码（为了增强容错性，应使得编码间的最小汉明距离尽可能大）。

1.9杰卡德相似系数(Jaccardsimilarity coefficient)

(1) 杰卡德相似系数

两个集合A和B的交集元素在A，B的并集中所占的比例，称为两个集合的杰卡德相似系数，用符号J(A,B)表示。

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/20.png">

杰卡德相似系数是衡量两个集合的相似度一种指标。

(2) 杰卡德距离

与杰卡德相似系数相反的概念是杰卡德距离(Jaccarddistance)。

杰卡德距离可用如下公式表示：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/21.png">

杰卡德距离用两个集合中不同元素占所有元素的比例来衡量两个集合的区分度。

(3)杰卡德相似系数与杰卡德距离的应用

可将杰卡德相似系数用在衡量样本的相似度上。

样本A与样本B是两个n维向量，而且所有维度的取值都是0或1。例如：A(0111)和B(1011)。我们将样本看成是一个集合，1表示集合包含该元素，0表示集合不包含该元素。

p：样本A与B都是1的维度的个数

q：样本A是1，样本B是0的维度的个数

r：样本A是0，样本B是1的维度的个数

s：样本A与B都是0的维度的个数

那么样本A与B的杰卡德相似系数可以表示为：

这里p+q+r可理解为A与B的并集的元素个数，而p是A与B的交集的元素个数。

而样本A与B的杰卡德距离表示为：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/22.png">

1.10相关系数( Correlation coefficient )与相关距离(Correlation distance)

(1)相关系数的定义

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/23.png">

相关系数是衡量随机变量X与Y相关程度的一种方法，相关系数的取值范围是[-1,1]。相关系数的绝对值越大，则表明X与Y相关度越高。当X与Y线性相关时，相关系数取值为1（正线性相关）或-1（负线性相关）。

(2)相关距离的定义

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/24.png">

