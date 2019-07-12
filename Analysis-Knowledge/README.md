# 分析/统计/数挖笔记

**1.距离度量 & 相似性度量(distance & similarity measurement)**

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

1.1欧氏距离(EuclideanDistance)
欧氏距离是最易于理解的一种距离计算方法，源自欧氏空间中两点间的距离公式。

(1)二维平面上两点a(x1,y1)与b(x2,y2)间的欧氏距离：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/1.png">

(2)三维空间两点a(x1,y1,z1)与b(x2,y2,z2)间的欧氏距离：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/2.png">

(3)两个n维向量a(x11,x12,…,x1n)与 b(x21,x22,…,x2n)间的欧氏距离：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/3.png">

也可以用表示成向量运算的形式：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/4.png">

1.2曼哈顿距离(ManhattanDistance)
想象你在曼哈顿要从一个十字路口开车到另外一个十字路口，驾驶距离是两点间的直线距离吗？显然不是，除非你能穿越大楼。实际驾驶距离是你走过马路距离，而曼哈顿的马路均为直线(十字路口)，就是这个“曼哈顿距离”。而这也是曼哈顿距离名称的来源， 曼哈顿距离也称为城市街区距离(CityBlock distance)。

举个栗子：点和橙点曼哈顿距离 = 穿过的横竖直线和

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/5.png" width="300">

(1)二维平面两点a(x1,y1)与b(x2,y2)间的曼哈顿距离

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/6.png">

(2)两个n维向量a(x11,x12,…,x1n)与b(x21,x22,…,x2n)间的曼哈顿距离

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/7.png">

1.3切比雪夫距离 ( Chebyshev Distance )
假设平面空间有2点，分别为(x1,y1)，(x2,y2)。两点的横纵坐标差的最大值max(| x2-x1 | , | y2-y1 | ) ，即切比雪夫距离。

(1)二维平面两点a(x1,y1)与b(x2,y2)间的切比雪夫距离

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/8.png">

(2)两个n维向量a(x11,x12,…,x1n)与b(x21,x22,…,x2n)间的切比雪夫距离

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/9.png">

另一种等价形式是

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/10.png">

1.4闵可夫斯基距离(MinkowskiDistance)

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

1.5标准化欧氏距离(Standardized Euclidean distance )

(1)标准欧氏距离的定义

标准化欧氏距离是针对简单欧氏距离的缺点而作的一种改进方案。标准欧氏距离的思路：既然数据各维分量的分布不一样，好吧！那我先将各个分量都“标准化”到均值、方差相等吧。均值和方差标准化到多少呢？这里先复习点统计学知识吧，假设样本集X的均值(mean)为m，标准差(standarddeviation)为s，那么X的“标准化变量”表示为

标准化变量的数学期望为0，方差为1。因此样本集的标准化过程(standardization)用公式描述就是：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/12.png">

标准化后的值 =  ( 标准化前的值  － 分量的均值 ) /分量的标准差

经过简单的推导就可以得到两个n维向量a(x11,x12,…,x1n)与b(x21,x22,…,x2n)间的标准化欧氏距离的公式：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Analysis-Knowledge/images/13.png">

如果将方差的倒数看成是一个权重，这个公式可以看成是一种加权欧氏距离(WeightedEuclidean distance)。



