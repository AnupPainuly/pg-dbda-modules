import pandas as pd 
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor 

import matplotlib.pyplot as plt
import numpy as np 
from sklearn.ensemble import VotingRegressor

concrete = pd.read_csv("Concrete_Data.csv")
X = concrete.drop('Strength', axis=1)
y = concrete['Strength']

lr = LinearRegression()
ridge = Ridge()
lasso = Lasso()
dtr = DecisionTreeRegressor(random_state=23) 
kfold = KFold(n_splits=5, shuffle=True, random_state=23)

def kfold_cv(model):
    res = cross_val_score(model, X, y,
                          scoring='r2', cv=kfold)
    return res.mean()

print("Linear Regression =", kfold_cv(lr))
print("Ridge Regression =", kfold_cv(ridge))
print("Lasso Regression =", kfold_cv(lasso))
print("Decision Tree Regression =", kfold_cv(dtr))

voting = VotingRegressor([('LR',lr),
                          ('RIDGE',ridge),
                          ('LASSO',lasso),
                          ('TREE',dtr)],
                         weights=[0.6, 0.6, 0.6, 0.8])

