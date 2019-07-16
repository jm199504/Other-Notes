## 各种小抄

### 1.GPU运行python代码

进入虚拟环境：

```javascript
source activate python36
```

查看空闲

```javascript
GPU：nvidia-smi
```

指定GPU运行代码（运行）：

```javascript
CUDA_VISIBLE_DEVICES=0,1,2 python test.py
```

指定GPU运行代码（代码）：

```javascript
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
```

监视GPU运行状态（10秒更新）：

```javascript
watch -n 10 nvidia-smi
```

### 2.读取语料库（自然语言处理）

#### 2.1寻找相似度高的前五个词语

```javascript
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin")
sim = model.most_similarity(postive=['woman','king'],negative=['man'],topn=5)
print(sim)
print(len(model['girl']))
```

#### 2.2 语料库下载

中文wiki语料库(1.2G)

<https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2>

英文wiki语料库(11.9G)

https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

### 3.运行时间

```javascript
import time
start = time.clock()
elapsed = (time.clock()-start)
print("time used:",elapsed)
```