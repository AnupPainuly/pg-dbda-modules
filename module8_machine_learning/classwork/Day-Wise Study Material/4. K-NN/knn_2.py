import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler 

concrete = pd.read_csv("Concrete_Data.csv")
X = concrete.drop('Strength', axis=1)
y = concrete['Strength']

### Standard Scaler
scaler = StandardScaler()
knn = KNeighborsRegressor()
params = {'KNN__n_neighbors': np.arange(1,31)}
pipe = Pipeline([('SCL',scaler),('KNN',knn)])
kfold = KFold(n_splits=5, shuffle=True, random_state=23)
gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   scoring='neg_mean_squared_error',
                   cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)


### MinMax Scaler
scaler = MinMaxScaler()
knn = KNeighborsRegressor()
params = {'KNN__n_neighbors': np.arange(1,31)}
pipe = Pipeline([('SCL',scaler),('KNN',knn)])
kfold = KFold(n_splits=5, shuffle=True, random_state=23)
gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   scoring='neg_mean_squared_error',
                   cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

#################Carseats########################
from ISLP import load_data
Carseats = load_data('Carseats')
Carseats.info()
dum_seats = pd.get_dummies(Carseats, drop_first=True)
X = dum_seats.drop('Sales', axis=1)
y = dum_seats['Sales']
kfold = KFold(n_splits=5, shuffle=True, random_state=23)
#### Linear Regression
lr = LinearRegression()
results = cross_val_score(lr, X, y, scoring='r2',
                          cv=kfold)
print(results.mean())
eval_df = pd.DataFrame({'Model':["Linear Regression"],
                        'Params':[''],
                        'Score':[results.mean()]})

#### Ridge
ridge = Ridge()
params = {'alpha':np.linspace(0.001,10, 30)}
gcv = GridSearchCV(ridge, param_grid=params,
                   cv=kfold, scoring='r2')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)
eval_df.loc[len(eval_df.index)] = ['Ridge',gcv.best_params_,
                                   gcv.best_score_]

#### Lasso
lasso = Lasso()
params = {'alpha':np.linspace(0.001,10, 30)}
gcv = GridSearchCV(lasso, param_grid=params,
                   cv=kfold, scoring='r2')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)
eval_df.loc[len(eval_df.index)] = ['Lasso',gcv.best_params_,
                                   gcv.best_score_]

#### ElasticNet
elastic = ElasticNet()
params = {'alpha':np.linspace(0.001,10, 30),
          'l1_ratio':np.linspace(0.001,0.999,10)}
gcv = GridSearchCV(elastic, param_grid=params,
                   cv=kfold, scoring='r2')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)
eval_df.loc[len(eval_df.index)] = ['Elastic Net',gcv.best_params_,
                                   gcv.best_score_]
#### K-NN
scaler = StandardScaler()
knn = KNeighborsRegressor()
params = {'KNN__n_neighbors': np.arange(1,36)}
pipe = Pipeline([('SCL',scaler),('KNN',knn)])
kfold = KFold(n_splits=5, shuffle=True, random_state=23)
gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   scoring='r2',
                   cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
eval_df.loc[len(eval_df.index)] = ['K-NN',gcv.best_params_,  gcv.best_score_]
     
eval_df.sort_values(by="Score", ascending=False)                                 
