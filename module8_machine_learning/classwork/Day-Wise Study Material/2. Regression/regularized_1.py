import pandas as pd 
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

boston = pd.read_csv("Boston.csv")

X = boston.drop('medv', axis=1)
y = boston['medv']

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3,
                                                    random_state=23)

ridge = Ridge(alpha=1)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
print(mean_squared_error(y_test, y_pred))

############ Grid Search CV #######################
kfold = KFold(n_splits=5, shuffle=True, random_state=23)
ridge = Ridge()
print(ridge.get_params())
params = {'alpha': np.linspace(0.001, 15)}
gcv = GridSearchCV(ridge, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_mean_squared_error')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)



############ Randomized Grid Search CV #######################
from sklearn.model_selection import RandomizedSearchCV
kfold = KFold(n_splits=5, shuffle=True, random_state=23)
ridge = Ridge()
print(ridge.get_params())
params = {'alpha': np.linspace(0.001, 15, 100)}
rgcv = RandomizedSearchCV(ridge, param_distributions=params,
                          verbose=3,n_iter=20, random_state=23,
                   cv=kfold, scoring='neg_mean_squared_error')
rgcv.fit(X, y)
print(rgcv.best_params_)
print(rgcv.best_score_)

# rgcv_res = pd.DataFrame(rgcv.cv_results_)
# rgcv_res.to_csv("GridResults.csv")

################### ElasticNet ###########################
from sklearn.linear_model import ElasticNet
kfold = KFold(n_splits=5, shuffle=True, random_state=23)
elastic = ElasticNet()
print(elastic.get_params())
params = {'alpha': np.linspace(0.001, 15,20),
          'l1_ratio': np.linspace(0.001, 0.999, 25)}
gcv = GridSearchCV(elastic, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_mean_squared_error')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
