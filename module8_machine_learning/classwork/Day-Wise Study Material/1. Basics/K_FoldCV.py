import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import r2_score

houses = pd.read_csv("Housing.csv")
dum_houses = pd.get_dummies(houses, drop_first=True)

X = dum_houses.drop('price', axis=1)
y = dum_houses['price']

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    random_state=23,
                                                    test_size=0.3)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(r2_score(y_test, y_pred))


scores = []
depth_values = [1,2,3,4,5,6,7,8,9,10]
for depth in depth_values:
    dtr = DecisionTreeRegressor(max_depth=5,random_state=23)
    dtr.fit(X_train, y_train)
    ycap = dtr.predict(X_test)
    scores.append(r2_score(y_test, ycap))

i_max = np.argmax(scores)
print("Best Depth =", depth_values[i_max])
print("Best Score =",scores[i_max])

#####################K-Fold CV############################
from sklearn.model_selection import KFold, cross_val_score
######## Linear Regression
lr = LinearRegression()
kfold = KFold(n_splits=5, shuffle=True, random_state=23)
results = cross_val_score(lr, X, y, cv=kfold)
print(results.mean())

###### Tree
kfold = KFold(n_splits=5, shuffle=True, random_state=23)
dtr = DecisionTreeRegressor(max_depth=5,random_state=23)
results = cross_val_score(dtr, X, y, cv=kfold)
print(results.mean())


scores = []
depth_values = [1,2,3,4,5,6,7,8,9,10]
for depth in depth_values:
    dtr = DecisionTreeRegressor(max_depth=depth,random_state=23)
    results = cross_val_score(dtr, X, y, cv=kfold)
    scores.append(results.mean())

i_max = np.argmax(scores)
print("Best Depth =", depth_values[i_max])
print("Best Score =",scores[i_max])


for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
    print("Fold", i+1)
    print("Train:", len(train_index))
    print("Test:", len(test_index))
    
################ Grid Search CV #############
from sklearn.model_selection import GridSearchCV
depth_values = [1,2,3,4,5,6,7,8,9,10]
dtr = DecisionTreeRegressor(random_state=23)
params = {'max_depth': depth_values}
print(dtr.get_params())

gcv = GridSearchCV(dtr, param_grid=params,
                   cv=kfold, 
                   scoring='neg_mean_squared_error')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

df_results = pd.DataFrame(gcv.cv_results_)
df_results.to_csv("GridResults.csv",
                  index=False)






