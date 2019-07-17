## 数据分析

### 1.绘图库

import seaborn as sns
import matplotlib.pyplot as plt

### 2.分析某列数据的分布及可视化

```javascript
# 标签值分布可视化
sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(8, 7))

sns.distplot(train['SalePrice'], color="b");
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="SalePrice distribution")
sns.despine(trim=True, left=True)
plt.show()
```
<img src="https://github.com/jm199504/Other-Notes/blob/master/Data-Analysis/House%20Prices%20Advanced%20Regression%20Techniques/images/1.png">

### 3.分析某列数据的偏度和峰度

偏度（Skewness）是描述数据分布形态的统计量，其描述的是某总体取值分布的对称性，简单来说就是数据的不对称程度。。
偏度是三阶中心距计算出来的。

（1）Skewness = 0 ，分布形态与正态分布偏度相同。

（2）Skewness > 0 ，正偏差数值较大，为正偏或右偏。长尾巴拖在右边，数据右端有较多的极端值。

（3）Skewness < 0 ，负偏差数值较大，为负偏或左偏。长尾巴拖在左边，数据左端有较多的极端值。

（4）数值的绝对值越大，表明数据分布越不对称，偏斜程度大。

| Skewness| 越大，分布形态偏移程度越大。

峰度（Kurtosis）偏度是描述某变量所有取值分布形态陡缓程度的统计量，简单来说就是数据分布顶的尖锐程度。

峰度是四阶标准矩计算出来的。

（1）Kurtosis=0 与正态分布的陡缓程度相同。

（2）Kurtosis>0 比正态分布的高峰更加陡峭——尖顶峰

（3）Kurtosis<0 比正态分布的高峰来得平台——平顶峰

```javascript
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())
```

### 4.分析各列数据的相关性

```javascript
# 分析数据特征间相关性
corr = train.corr()
plt.subplots(figsize=(15,12))
sns.heatmap(corr, vmax=0.9, cmap="Blues", square=True)
```

<img src="https://github.com/jm199504/Other-Notes/blob/master/Data-Analysis/House%20Prices%20Advanced%20Regression%20Techniques/images/2.png" width="500">

### 5.提取数值型数据特征

```javascript
# 寻找数值型特征
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in train.columns:
    if train[i].dtype in numeric_dtypes:
        numeric.append(i)
```

### 6.分析各列数据与标签值的相关性（散点图可视化）

```javascript
# 新建画布
fig, axs = plt.subplots(ncols=2, nrows=0, figsize=(12, 120))
plt.subplots_adjust(right=2)
plt.subplots_adjust(top=2)
sns.color_palette("husl", 8)
# 生成数据特征与销售价格SalePrice的相关散点图（每行3图）
for i, feature in enumerate(list(train[numeric]), 1):
    plt.subplot(len(list(numeric)), 3, i)
    sns.scatterplot(x=feature, y='SalePrice', hue='SalePrice', palette='Blues', data=train)
    # 绘制x与y坐标轴名    
    plt.xlabel('{}'.format(feature), size=15,labelpad=12.5)
    plt.ylabel('SalePrice', size=15, labelpad=12.5)
    # 绘制x与y坐标轴
    for j in range(2):
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)

    plt.legend(loc='best', prop={'size': 10})

plt.show()
```

<img src="https://github.com/jm199504/Other-Notes/blob/master/Data-Analysis/House%20Prices%20Advanced%20Regression%20Techniques/images/3.png">

### 7.分析各列数据与标签值的相关性（箱体图可视化）

```javascript
# 查看具体的数据特征（OverallQual）与SalePrice 的相关性
data = pd.concat([train['SalePrice'], train['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=train['OverallQual'], y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
```

<img src="https://github.com/jm199504/Other-Notes/blob/master/Data-Analysis/House%20Prices%20Advanced%20Regression%20Techniques/images/4.png">

### 8.对非正态分布的标签值进行修正

大多数机器学习模型对非正态分布的数据的效果不佳，因此考虑对数据进行变换/修正倾斜：log(1+x) 

```javascript
train["SalePrice"] = np.log1p(train["SalePrice"])
```

### 9.删除异常数据

```javascript
# 删除异常值
train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index, inplace=True)
train.drop(train[(train['GrLivArea']>4500) & (train['SalePrice']<300000)].index, inplace=True)
train.reset_index(drop=True, inplace=True)
```

### 10.合并训练集和测试集的数据特征，分裂训练集的标签列

```javascript
# 合并训练集和测试集，并分割数据特征与标签
train_labels = train['SalePrice'].reset_index(drop=True)
train_features = train.drop(['SalePrice'], axis=1)
test_features = test
# Combine train and test features in order to apply the feature transformation pipeline to the entire dataset
all_features = pd.concat([train_features, test_features]).reset_index(drop=True)
all_features.shape
```

### 11.统计各列数据的缺失值比例

```javascript
# 统计缺失值比例
def percent_missing(df):
    data = pd.DataFrame(df)
    df_cols = list(pd.DataFrame(data))
    dict_x = {}
    for i in range(0, len(df_cols)):
        dict_x.update({df_cols[i]: round(data[df_cols[i]].isnull().mean()*100,2)})
    return dict_x

missing = percent_missing(all_features)
df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)
print('Percent of missing data')
df_miss[0:10]
```

### 12.可视化各列数据缺失值比例

```javascript
# 可视化缺失值比率
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
sns.set_color_codes(palette='deep')
missing = round(train.isnull().mean()*100,2)
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar(color="b")

ax.xaxis.grid(False)
ax.set(ylabel="Percent of missing values")
ax.set(xlabel="Features")
ax.set(title="Percent missing data by feature")
sns.despine(trim=True, left=True)
```

<img src="https://github.com/jm199504/Other-Notes/blob/master/Data-Analysis/House%20Prices%20Advanced%20Regression%20Techniques/images/5.png">

### 13.填补缺失值

#### 13.1 众数

```javascript
features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
```

其中.mode()[0]表示该列众数，.mode()[1]表示该行众数

#### 13.2 中位数（分组）

```javascript
features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
```

### 14.分析各列数据的分布（箱体图可视化）

```javascript
# 可视化数值特征
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
ax.set_xscale("log")
ax = sns.boxplot(data=all_features[numeric] , orient="h", palette="Set1")
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features")
sns.despine(trim=True, left=True)
```

<img src="https://github.com/jm199504/Other-Notes/blob/master/Data-Analysis/House%20Prices%20Advanced%20Regression%20Techniques/images/6.png">

### 15.提取倾斜数据

```javascript
# 找到倾斜的数值特征
skew_features = all_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))
skewness = pd.DataFrame({'Skew' :high_skew})
skew_features.head(10)
```

### 16.使用Box-cox对倾斜数据转换成正态分布

```javascript
# 使用 scipy 的函数 boxcox1来进行 Box-Cox 转换，将数据正态化
for i in skew_index:
    all_features[i] = boxcox1p(all_features[i],boxcox_normmax(all_features[i] + 1))
```

```javascript
# 检查数据特征是否均满足正态分布（箱体图可视化）
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
ax.set_xscale("log")
ax = sns.boxplot(data=all_features[skew_index] , orient="h", palette="Set1")
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features")
sns.despine(trim=True, left=True)
```

<img src="https://github.com/jm199504/Other-Notes/blob/master/Data-Analysis/House%20Prices%20Advanced%20Regression%20Techniques/images/7.png">

### 17.产生新的数据特征

```javascript
# 加权求和数据特征
all_features['Total_Bathrooms'] = (all_features['FullBath'] + (0.5 * all_features['HalfBath']) +all_features['BsmtFullBath'] + (0.5 * all_features['BsmtHalfBath']))
# 类别数据特征衍生（无该特征作为新的数据特征）
all_features['HasWoodDeck'] = (all_features['WoodDeckSF'] == 0) * 1
# 指数生成新的数据特征
all_features['TotalBsmtSF'] = all_features['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
# 所有数据特征经过对数转换:通过对特征取对数，创造更多的特征，有利于发掘潜在的有用特征
def logs(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(np.log(1.01+res[l])).values)
        res.columns.values[m] = l + '_log'
        m += 1
    return res
## 所有数据特征经过平方转换
def squares(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(res[l]*res[l]).values)
        res.columns.values[m] = l + '_sq'
        m += 1
    return res
```

### 18.对类别数据特征进行哑变量转换

```javascript
# 对集合特征进行数值编码，使得机器学习模型能够处理这些特征。
all_features = pd.get_dummies(all_features).reset_index(drop=True)
all_features.shape
```

### 19.去除重复数据特征

```javascript
# 去除重复特征
all_features = all_features.loc[:,~all_features.columns. duplicated()]
all_features.shape
```

### 20.重新分割回训练集和测试集

```javascript
# 重新获得训练集和测试集
X = all_features.iloc[:len(train_labels), :]
X_test = all_features.iloc[len(train_labels):, :]
X.shape, train_labels.shape, X_test.shape
```

### 21.初始化交叉验证和定义误差评估指标

```javascript
# 初始化交叉验证
kf = KFold(n_splits=12, random_state=42, shuffle=True)

# 定义误差评估指标
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, train_labels, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)
```

### 22.构建预测模型

#### 22.1 Light Gradient Boosting Regressor

#### 22.2 XGBoost Regressor

#### 22.3 Ridge Regressor

#### 22.4 Support Vector Regressor

#### 22.5 Gradient Boosting Regressor

#### 22.6 Random Forest Regressor

#### 22.7 Stacking

```javascript
# 构建模型
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

### 23.计算每个模型的交叉验证的得分

```javascript
# lightgbm
scores = {}
score = cv_rmse(lightgbm)
print("lightgbm: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['lgb'] = (score.mean(), score.std())

# xgboost
score = cv_rmse(xgboost)
print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['xgb'] = (score.mean(), score.std())

# svr
score = cv_rmse(svr)
print("SVR: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['svr'] = (score.mean(), score.std())

# ridge
score = cv_rmse(ridge)
print("ridge: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['ridge'] = (score.mean(), score.std())

# rf
score = cv_rmse(rf)
print("rf: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['rf'] = (score.mean(), score.std())

# gbr
score = cv_rmse(gbr)
print("gbr: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['gbr'] = (score.mean(), score.std())
```

### 24.拟合模型

```javascript
# 拟合模型
print('stack_gen')
stack_gen_model = stack_gen.fit(np.array(X), np.array(train_labels))
```

### 25.混合模型

```javascript
# 混合模型，以使最终的预测更健壮，以过度拟合
def blended_predictions(X):
    return ((0.1 * ridge_model_full_data.predict(X)) + \
            (0.2 * svr_model_full_data.predict(X)) + \
            (0.1 * gbr_model_full_data.predict(X)) + \
            (0.1 * xgb_model_full_data.predict(X)) + \
            (0.1 * lgb_model_full_data.predict(X)) + \
            (0.05 * rf_model_full_data.predict(X)) + \
            (0.35 * stack_gen_model.predict(np.array(X))))
```

### 26.绘制各模型得分

```javascript
sns.set_style("white")
fig = plt.figure(figsize=(24, 12))

ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()], markers=['o'], linestyles=['-'])
for i, score in enumerate(scores.values()):
    ax.text(i, score[0] + 0.002, '{:.6f}'.format(score[0]), horizontalalignment='left', size='large', color='black', weight='semibold')

plt.ylabel('Score (RMSE)', size=20, labelpad=12.5)
plt.xlabel('Model', size=20, labelpad=12.5)
plt.tick_params(axis='x', labelsize=13.5)
plt.tick_params(axis='y', labelsize=12.5)

plt.title('Scores of Models', size=20)

plt.show
```

<img src="https://github.com/jm199504/Other-Notes/blob/master/Data-Analysis/House%20Prices%20Advanced%20Regression%20Techniques/images/8.png" width="500">

### 27.读取提交示范代码格式

```javascript
submission = pd.read_csv("house-prices-advanced-regression-techniques/sample_submission.csv")
submission.shape
```

### 28.调整预测结果

```javascript
# 调整混合模型的预测
submission.iloc[:,1] = np.floor(np.expm1(blended_predictions(X_test)))

# 修复异常点的预测值
q1 = submission['SalePrice'].quantile(0.0045)
q2 = submission['SalePrice'].quantile(0.99)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
submission.to_csv("submission_regression1.csv", index=False)

# 比例缩放
submission['SalePrice'] *= 1.001619
submission.to_csv("submission_regression2.csv", index=False)
```

