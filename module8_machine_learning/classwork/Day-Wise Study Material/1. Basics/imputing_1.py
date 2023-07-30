import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn.impute import SimpleImputer 
##############################################
a = np.array([[2,7,45],
              [4,3,33],
              [np.nan, 1, np.nan],
              [6,np.nan, 56],
              [	np.nan, 7,67],
              [5,4,87]])

imputer = SimpleImputer(strategy='mean') 
imputer.fit(a)
print(imputer.statistics_)
imputer.transform(a)

imputer.fit_transform(a)
#############################################

chem = pd.read_csv("ChemicalProcess.csv")
print(chem.isnull().sum())

X = chem.drop('Yield', axis=1)
y = chem['Yield']

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3,
                                                    random_state=23)
imputer = SimpleImputer() 
imp_X_train = imputer.fit_transform(X_train)
imp_X_test = imputer.transform(X_test)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(imp_X_train)
X_test_poly = poly.transform(imp_X_test)

lr = LinearRegression()
lr.fit(X_train_poly, y_train)
y_pred = lr.predict(X_test_poly)
print(mean_squared_error(y_test, y_pred))


################# Pipeline ########################
from sklearn.pipeline import Pipeline 
imputer = SimpleImputer() 
poly = PolynomialFeatures(degree=2)
lr = LinearRegression()
pipe = Pipeline([('IMPUTER',imputer),
                 ('POLY',poly),('LR',lr)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(mean_squared_error(y_test, y_pred))

########### Grid Search with Pipeline ################
kfold = KFold(n_splits=5, shuffle=True, random_state=23)
imputer = SimpleImputer() 
poly = PolynomialFeatures()
lr = LinearRegression()
pipe = Pipeline([('IMPUTER',imputer),
                 ('POLY',poly),('LR',lr)])
print(pipe.get_params())
params = {'IMPUTER__strategy':['mean','median'],
          'POLY__degree':[1,2,3]}
gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_mean_squared_error')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

gcv_res = pd.DataFrame(gcv.cv_results_)
gcv_res.to_csv("GridResults.csv")
