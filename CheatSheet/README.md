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

### 4.预测竞赛

#### 4.1 注意事项

1.明确竞赛评价指标（损失函数）

2.交叉验证：使用12-折交叉验证

3.模型融合Stacking（ridge, svr, gradient boosting, random forest, xgboost, lightgbm regressors）

#### 4.2 流程

1.探索各数据特征与标签值的相关系数（corr）

2.对数据特征可视化分析（数据特征值分布）：若非正态分布，考虑使用对数变换、Box-cox变换

from scipy.stats import skew, norm

from scipy.special import boxcox1p

3.填充缺失值（分析各列缺失值量和具体特征意义采取不同方式填充）

4.特征衍生（特征加权求和，0-1离散化，取对数，取平方）

5.离散（分类）特征进行哑变量转换（某列有3个类则衍生3个特征其分别为0/1）

6.计算每个模型的交叉验证的得分，基于验证集的拟合效果，对模型结果加权求和



常用预测模型（参考）：

```javascript
# Light Gradient Boosting Regressor
lightgbm = LGBMRegressor(objective='regression',
                       num_leaves=6,
                       learning_rate=0.01,
                       n_estimators=7000,
                       max_bin=200,
                       bagging_fraction=0.8,
                       bagging_freq=4,
                       bagging_seed=8,
                       feature_fraction=0.2,
                       feature_fraction_seed=8,
                       min_sum_hessian_in_leaf = 11,
                       verbose=-1,
                       random_state=42)

# XGBoost Regressor
xgboost = XGBRegressor(learning_rate=0.01,
                       n_estimators=6000,
                       max_depth=4,
                       min_child_weight=0,
                       gamma=0.6,
                       subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:linear',
                       nthread=-1,
                       scale_pos_weight=1,
                       seed=27,
                       reg_alpha=0.00006,
                       random_state=42)


# Ridge Regressor
ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))


# Support Vector Regressor
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))


# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=6000,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=42)


# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=1200,
                          max_depth=15,
                          min_samples_split=5,
                          min_samples_leaf=5,
                          max_features=None,
                          oob_score=True,
                          random_state=42)


# Stack up all the models above, optimized using xgboost
stack_gen = StackingCVRegressor(regressors=(xgboost, lightgbm, svr, ridge, gbr, rf),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)
```

### 5.忽略无用的警告

```
# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")
```

### 6.Skleanr自带的数据库
```
from sklearn import datasets, model_selection, naive_bayes

def load_data(datasets_name='iris'):
    if datasets_name == 'iris':
        data = datasets.load_iris()  # 加载 scikit-learn 自带的 iris 鸢尾花数据集-分类
    elif datasets_name == 'wine': # 0.18.2 没有
        data = datasets.load_wine()  # 加载 scikit-learn 自带的 wine 红酒起源数据集-分类
    elif datasets_name == 'cancer':
        data = datasets.load_breast_cancer()  # 加载 scikit-learn 自带的 乳腺癌数据集-分类
    elif datasets_name == 'digits':
        data = datasets.load_digits()  # 加载 scikit-learn 自带的 digits 糖尿病数据集-回归
    elif datasets_name == 'boston':
        data = datasets.load_boston()  # 加载 scikit-learn 自带的 boston 波士顿房价数据集-回归
    else:
        pass
    return model_selection.train_test_split(data.data, data.target,test_size=0.25, random_state=0,stratify=data.target)
 # 分层采样拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4

dataset = load_data('iris')

print(datasets)
```
