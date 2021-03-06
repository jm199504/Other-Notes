## 激活函数

函数目录

>1. Sigmoid
>2. Tanh
>3. Relu
>4. LeakyReLu
>5. LReLU 与 PReLU
>6. ELU

**1.Sigmoid**

是使用范围最广的一类激活函数，具有指数函数形状 。正式定义为: 

<img src="https://github.com/jm199504/Other-Notes/blob/master/Activation-Function/images/1.png" width="200">

求导：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Activation-Function/images/2.png" width="300">

当z = 10或z = -10，g’(z) ≈0

当z = 0，g’(z) = g(z) (1- g(z)) = 1/4

所以，当z≥10，或z≤-10，sigmoid函数的导数接近于0，在梯度下降算法中，反向传播很容易出现梯度消失的情况，造成无法完成深层神经网络的训练。

函数图像：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Activation-Function/images/3.png" width="300">

可见，sigmoid在定义域内处处可导，同时也被定义为软饱和激活函数。

一般来说， sigmoid 网络在 5 层之内就会产生梯度消失现象。

Sigmoid 函数能将负无穷到正无穷的数映射到0和1之间，并且对这个函数求导的结果是 f′(x)=f(x)(1−f(x))。因此两个0到1之间的数相乘，得到的结果就会变得很小了。

神经网络的反向传播是逐层对函数偏导相乘，因此当神经网络层数非常深的时候，最后一层产生的偏差就因为乘了很多的小于1的数而越来越小，最终就会变为0，从而导致层数比较浅的权重没有更新。

优点：Sigmoid 函数的输出映射在(0,1)之间，单调连续，输出范围有限，优化稳定，可以用作输出层。它在物理意义上最为接近生物神经元；求导容易。

缺点：由于其软饱和性，容易产生梯度消失，导致训练出现问题；其输出并不是以0为中心的。

**2.Tanh**

定义：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Activation-Function/images/4.png" width="200">

求导：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Activation-Function/images/5.png" width="300">

当z = 10或z = -10，g’(z) ≈0

当z = 0，g’(z) = g(z) (1- g(z)) = 1

与sigmoid函数一样，当z≥10，或z≤-10，tanh函数的导数接近于0，在梯度下降算法中，反向传播很容易出现梯度消失的现象，造成无法完成深层神经网络的训练。

函数位于[-1, 1]区间上，对应的图像是：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Activation-Function/images/6.png" width="300">

同样地，Tanh 激活函数也具有软饱和性。

Tanh 网络的收敛速度要比 Sigmoid 快，因为 Tanh 的输出均值比 Sigmoid 更接近 0，SGD 会更接近 natural gradient（一种二次优化技术），从而降低所需的迭代次数。

优点：比Sigmoid函数收敛速度更快，相比Sigmoid函数，其输出以0为中心。

缺点：没有改变Sigmoid函数的最大问题——由于饱和性产生的梯度消失。

**3.ReLU（隐藏层标配）**

定义：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Activation-Function/images/7.png" width="200">

函数图像：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Activation-Function/images/8.png" width="300">

可见，ReLU 在x<0 时硬饱和。由于 x>0时导数为 1，所以，ReLU 能够在x>0时保持梯度不衰减，从而缓解梯度消失问题。但随着训练的推进，部分输入会落入硬饱和区导致对应权重无法更新，被称为“神经元死亡”。

优点：

1.相比起Sigmoid和tanh，ReLU在SGD中能够快速收敛（线性、非饱和的形式）。

2.Sigmoid和tanh涉及了很多很expensive的操作（比如指数），ReLU实现简单。

3.有效缓解了梯度消失的问题。

4.提供了神经网络的稀疏表达能力。

缺点：部分输入可能落入硬饱和区导致对应权重无法更新，“神经元死亡”。

**4.LeakyReLu**

定义：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Activation-Function/images/9.png" width="200">

函数图像：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Activation-Function/images/10.png" width="300">

**5.PReLU**

PReLU 是 ReLU 和 LReLU 的改进版本，具有非饱和性，定义：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Activation-Function/images/11.png" width="200">

函数图像：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Activation-Function/images/12.png" width="300">

**6.ELU**

ELU融合了sigmoid和ReLU，具有左侧软饱性。其正式定义为：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Activation-Function/images/13.png" width="300">

函数图像：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Activation-Function/images/14.png" width="300">

右侧线性部分使得ELU能够缓解梯度消失，而左侧软饱能够让ELU对输入变化或噪声更鲁棒。ELU的输出均值接近于零，所以收敛速度更快。

汇总：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Activation-Function/images/16.png">
