import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

house = pd.read_csv("Housing.csv")
dum_house = pd.get_dummies(house, drop_first=True)
X = dum_house.drop('price', axis=1)
y = dum_house['price']

clf = DecisionTreeRegressor(random_state=23)
params = {'max_depth':[None, 2, 3, 4, 5],
          'min_samples_split': np.arange(2,21,2),
          'min_samples_leaf': np.arange(1,21,2)}
kfold = KFold(n_splits=5, shuffle=True,
                        random_state=23)
gcv = GridSearchCV(clf, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_mean_squared_error')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)


best_model = gcv.best_estimator_

imps = best_model.feature_importances_
cols = list(X.columns)
imp_df = pd.DataFrame({'feature':cols,
                       'importance':imps})
imp_df.sort_values(by='importance', inplace=True)
plt.figure(figsize=(8,6))
plt.title("Sorted Importances Plot")
plt.barh(imp_df['feature'], imp_df['importance'])
plt.show()

################ California Housing #################
from sklearn.datasets import fetch_california_housing 

X, y = fetch_california_housing(as_frame=True, 
                               return_X_y=True)

clf = DecisionTreeRegressor(random_state=23)
params = {'max_depth':[None, 2, 3, 4, 5],
          'min_samples_split': np.arange(2,21,2),
          'min_samples_leaf': np.arange(1,21,2)}
kfold = KFold(n_splits=5, shuffle=True,
                        random_state=23)
gcv = GridSearchCV(clf, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_mean_squared_error')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)


best_model = gcv.best_estimator_

imps = best_model.feature_importances_
cols = list(X.columns)
imp_df = pd.DataFrame({'feature':cols,
                       'importance':imps})
imp_df.sort_values(by='importance', inplace=True)
plt.figure(figsize=(8,6))
plt.title("Sorted Importances Plot")
plt.barh(imp_df['feature'], imp_df['importance'])
plt.show()

################# Credit Card Balance ##############
from ISLP import load_data
Credit = load_data('Credit')
Credit.columns
