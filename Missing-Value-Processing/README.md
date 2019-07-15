## 缺失值处理



### 1.直接删除

data = data.replace(' ', np.NaN)

###  2.使用一个全局常量填充缺失值

\#均值填充

data['a'] = data['a'].fillna(data['a'].means())

\#中位数填充

data['a'] = data['a'].fillna(data['a'].median())

\#众数填充

data['a'] = data['a'].fillna(stats.mode(data['a'])[0][0])

\#用前一个数据进行填充

data['a'] = data['a'].fillna(method='pad')

\#用后一个数据进行填充

data['a'] = data['a'].fillna(method='bfill')

附加：Imputer提供了缺失数值处理的基本策略，比如使用缺失数值所在行或列的均值、中位数、众数来替代缺失值。

from sklearn.preprocessing import Imputer

imr = Imputer(missing_values='NaN', strategy='mean', axis=0)

imr = imr.fit(data.values)

imputed_data = pd.DataFrame(imr.transform(data.values))

参数可选：

<img src="https://github.com/jm199504/Other-Notes/blob/master/Missing-Value-Processing/images/1.png">

### 3.前后插值法

interpolate()插值法，计算的是缺失值前一个值和后一个值的平均数。

data['a'] = data['a'].interpolate()

### 4.KNN填充法

from fancyimpute import KNN

fill_knn = KNN(k=3).fit_transform(data)

data = pd.DataFrame(fill_knn)

print(data.head())

### 5.随机森林填充

from sklearn.ensemble import RandomForestRegressor

\#提取前五个数据特征

process_df = data.ix[:, [1, 2, 3, 4, 5]]

\# 分成已知该特征和未知该特征两部分

known = process_df[process_df.c.notnull()].as_matrix()

uknown = process_df[process_df.c.isnull()].as_matrix()

\# X为特征属性值

X = known[:, 1:3]

\# print(X[0:10])

\# Y为结果标签

y = known[:, 0]

print(y)

\# 训练模型

rf = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)

rf.fit(X, y)

\# 预测缺失值

predicted = rf.predict(uknown[:, 1:3])

print(predicted)

\#将预测值填补原缺失值

data.loc[(data.c.isnull()), 'c'] = predicted

print(data[0:10])
