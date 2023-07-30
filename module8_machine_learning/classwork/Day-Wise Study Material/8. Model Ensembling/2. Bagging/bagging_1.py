import pandas as pd 
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.tree import DecisionTreeRegressor

concrete = pd.read_csv("Concrete_Data.csv")
# =============================================================================
# 
# np.random.seed(23)
# ind_trn = np.random.randint(low=0, high=1029, size=1030)
# train = concrete.iloc[ind_trn,:]
# 
# # Bootstrap Sample
# d = np.arange(1,11)
# print(np.random.choice(d, 10, replace=True))
# 
# 
# np.random.seed(23)
# ind_trn = np.random.randint(low=0, high=1029, size=1030)
# repl_ind_trn = np.random.choice(ind_trn, len(ind_trn),
#                                 replace=True)
# train = concrete.iloc[repl_ind_trn,:]
# 
# =============================================================================

X = concrete.drop('Strength', axis=1)
y = concrete['Strength']

lr = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3,
                                                    random_state=23)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(r2_score(y_test, y_pred))

bagging = BaggingRegressor(estimator=lr,oob_score=True,
                           n_estimators=15, random_state=23)
bagging.fit(X_train, y_train)
print("Out of Bag Score =", bagging.oob_score_)

y_pred = bagging.predict(X_test)
print(r2_score(y_test, y_pred))

###############################################################
ridge = Ridge()
kfold = KFold(n_splits=5, shuffle=True, random_state=23)
results = cross_val_score(ridge, X, y, scoring='r2',
                          cv=kfold)
print(results.mean())

bagging = BaggingRegressor(estimator=ridge,
                           n_estimators=15, random_state=23)
results = cross_val_score(bagging, X, y, scoring='r2',
                          cv=kfold)
print(results.mean())

##### Grid Search CV
print(bagging.get_params())
params = {'estimator__alpha': np.linspace(0.001,5, 10)}
gcv = GridSearchCV(bagging, param_grid=params,
                   cv=kfold, scoring='r2')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

##### Tree
dtr = DecisionTreeRegressor(random_state=23)
bagging = BaggingRegressor(estimator=dtr,random_state=23,
                           n_jobs=-1)
print(bagging.get_params())
params = {'estimator__max_depth':[None, 2, 5, 6],
          'estimator__min_samples_split':[2, 5, 10, 20, 30, 40, 50],
          'estimator__min_samples_leaf':[1,5, 10, 20, 30, 40, 50],
          'n_estimators':[10, 25, 50]}
gcv = GridSearchCV(bagging, param_grid=params,n_jobs=-1,
                   cv=kfold, scoring='r2')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

########## multiple models
bagging = BaggingRegressor(random_state=23,
                           n_jobs=-1)
print(bagging.get_params())
params = {'estimator':[dtr, lr, ridge],
          'n_estimators':[10, 25, 50]}
gcv = GridSearchCV(bagging, param_grid=params,n_jobs=-1,
                   cv=kfold, scoring='r2')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
