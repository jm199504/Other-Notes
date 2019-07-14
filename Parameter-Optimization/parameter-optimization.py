# RandomizedSearchCV | 随机搜索
from sklearn.model_selection import RandomizedSearchCV
param_dist={"max_depth":[3,None],
            "max_features":sp_randint(1,11),
            "min_samples_split":sp_randint(2,11),
            "min_samples_leaf":sp_randint(1,11),
            "bootstrap":[True,False],
            "criterion":['gini','entropy']
            }
clf=RandomForestClassifier(n_estimators=20)
random_search=RandomizedSearchCV(clf,param_distributions=param_dist,n_iter=n_iter_search)
random_search.fit(X,y)


# GridSearchCV | 网格搜索
from sklearn.model_selection import GridSearchCV
model = LGBMRegressor()
param_grid = {'learning_rate':[.01, .05, .1, .5, .75, .9],'max_depth':[-1, 1, 3, 4, 5, 7, 9],
             'min_child_samples':[10, 15, 20, 25, 30],'min_child_weight':[.001, .01, .1, .5],
             'num_leaves':[25, 27, 29, 31, 33, 35, 37, 39, 41, 45, 50]}
gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
gs.fit(train_values_subset, train_labels.heart_disease_present)
print(gs.best_params_)


# BayesianOptimization | 贝叶斯超参优化
# pip install bayesian-optimization
from bayes_opt import BayesianOptimization
def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
    val = cross_val_score(
        RandomForestClassifier(n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999), # float
            max_depth=int(max_depth),
            random_state=2
        ),
        x, y, scoring='roc_auc', cv=5
    ).mean()
    return val
rf_bo = BayesianOptimization(
        rf_cv,
        {'n_estimators': (10, 250),
        'min_samples_split': (2, 25),
        'max_features': (0.1, 0.999),
        'max_depth': (5, 15)}
    )
print(rf_bo.maximize())#参数列表
maxindex = 0
maxvalue = 0
for i in range(len(rf_bo.res)):
    if rf_bo.res[i]['target']>maxvalue:
        maxvalue = rf_bo.res[i]['target']
        maxindex = i
print(rf_bo.res[maxindex]['target'])
print(rf_bo.res[maxindex]['params'])
gp_param={'kernel':None}
print(rf_bo.maximize(**gp_param))
rf = RandomForestClassifier(max_depth=11, max_features=0.5024, min_samples_split=9, n_estimators=146)
np.mean(cross_val_score(rf, x, y, cv=20, scoring='roc_auc'))
