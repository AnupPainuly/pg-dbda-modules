import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
pizza = pd.read_csv("pizza.csv")

X = pizza[['Promote']]
y = pizza['Sales']

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

lr = LinearRegression()
lr.fit(X_poly, y)

print("b0 =", lr.intercept_)
print("b's' =", lr.coef_)

#################### Boston ##############################
boston = pd.read_csv("Boston.csv")

X = boston[['dis','lstat']]
y = boston['medv']

poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)
print(poly.get_feature_names_out())

X_train, X_test, y_train, y_test = train_test_split(X_poly,y,
                                                    test_size=0.3,
                                                    random_state=23)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(mean_squared_error(y_test, y_pred))

##### with K-fold
kfold = KFold(n_splits=5, shuffle=True, random_state=23)
lr = LinearRegression()
degrees = [1,2,3,4,5]
scores = []
for n in degrees:
    poly = PolynomialFeatures(degree=n)
    X_poly = poly.fit_transform(X)
    results = cross_val_score(lr,X_poly,y, cv=kfold,
                              scoring='neg_mean_squared_error')
    scores.append(results.mean())
i_max = np.argmax(scores)
print("Best degree =", degrees[i_max])
print("Best Score =", scores[i_max])



################ taking all variables
kfold = KFold(n_splits=5, shuffle=True, random_state=23)
X = boston.drop('medv', axis=1)
y = boston['medv']
lr = LinearRegression()
degrees = [1,2,3,4,5]
scores = []
for n in degrees:
    poly = PolynomialFeatures(degree=n)
    X_poly = poly.fit_transform(X)
    results = cross_val_score(lr,X_poly,y, cv=kfold,
                              scoring='neg_mean_squared_error')
    scores.append(results.mean())
i_max = np.argmax(scores)
print("Best degree =", degrees[i_max])
print("Best Score =", scores[i_max])

##################### Pipeline #######################
from sklearn.pipeline import Pipeline

poly = PolynomialFeatures(degree=2)
lr = LinearRegression()

pipe = Pipeline([('POLY', poly),('LR', lr)])
results = cross_val_score(pipe,X,y, cv=kfold,
                          scoring='neg_mean_squared_error')
print(results.mean())

############# Grid Search with pipeline #########
from sklearn.model_selection import GridSearchCV
degrees = [1,2,3,4,5]
print(pipe.get_params())
params = {'POLY__degree':degrees}
gcv = GridSearchCV(pipe, param_grid=params,
                   cv=kfold,
                   scoring='neg_mean_squared_error')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)







