import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import StackingRegressor 
from sklearn.ensemble import RandomForestRegressor

concrete = pd.read_csv("Concrete_Data.csv")

X = concrete.drop('Strength', axis=1)
y = concrete['Strength']

lr = LinearRegression()
ridge = Ridge()
lasso = Lasso()
dtr = DecisionTreeRegressor(random_state=23)
models = [('LR', lr),('RIDGE',ridge ),('LASSO',lasso),('TREE',dtr)]
gbm = XGBClassifier(random_state=23)
cat_reg = CatBoostRegressor(random_state=23)
rf = RandomForestRegressor(random_state=23)
stack = StackingRegressor(estimators=models,final_estimator=rf,
                           passthrough=True)

####### Grid Search
print(stack.get_params())
kfold = KFold(n_splits=5, shuffle=True, random_state=23)
params = {'RIDGE__alpha':np.linspace(0.001, 5, 5),
          'LASSO__alpha':np.linspace(0.001, 5, 5),
          'TREE__max_depth':[None, 2,4],
          'TREE__min_samples_split':[2, 5, 10],
          'TREE__min_samples_leaf':[1, 5, 10],
          'final_estimator':[rf, gbm, cat_reg],
          'final_estimator__n_estimators':[50,100]}
rgcv = RandomizedSearchCV(stack, n_iter=20, n_jobs=-1,
                          param_distributions=params,
                   cv=kfold, scoring='r2')
rgcv.fit(X, y)
print(rgcv.best_params_)
print(rgcv.best_score_)
