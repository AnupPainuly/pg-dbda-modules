import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn import tree 
from sklearn.model_selection import train_test_split

############# Boston ##################
boston = pd.read_csv("Boston.csv")

train, test = train_test_split(boston, test_size=0.3,
                               random_state=23)

X_train = train.drop('medv', axis=1)
y_train = train['medv']
X_test = test.drop('medv', axis=1)
y_test = test['medv']

lr = LinearRegression()
lr.fit(X_train, y_train)
ycap = lr.predict(X_test)
print(mean_squared_error(y_test, ycap))


dtr = DecisionTreeRegressor(max_depth=3, random_state=23)
dtr.fit(X_train, y_train)

plt.figure(figsize=(15,10))
tree.plot_tree(dtr,feature_names=X_train.columns,
               filled=True,fontsize=10) 

ycap = dtr.predict(X_test)
print(mean_squared_error(y_test, ycap))

######## Concrete #########
concrete = pd.read_csv("Concrete_Data.csv")

train, test = train_test_split(concrete, test_size=0.3,
                               random_state=23)

X_train = train.drop('Strength', axis=1)
y_train = train['Strength']
X_test = test.drop('Strength', axis=1)
y_test = test['Strength']

lr = LinearRegression()
lr.fit(X_train, y_train)
ycap = lr.predict(X_test)
print(mean_squared_error(y_test, ycap))


dtr = DecisionTreeRegressor(max_depth=3, random_state=23)
dtr.fit(X_train, y_train)

plt.figure(figsize=(15,10))
tree.plot_tree(dtr,feature_names=X_train.columns,
               filled=True,fontsize=10) 

ycap = dtr.predict(X_test)
print(mean_squared_error(y_test, ycap))

errors = []
depth_values = [1,2,3,4,5,6,7,8,9,10]
for depth in depth_values:
    dtr = DecisionTreeRegressor(max_depth=depth, 
                                random_state=23)
    dtr.fit(X_train, y_train)
    ycap = dtr.predict(X_test)
    errors.append(mean_squared_error(y_test, ycap))

i_min = np.argmin(errors)
print("Best Depth =", depth_values[i_min])

dtr = DecisionTreeRegressor(max_depth=depth_values[i_min], 
                                random_state=23)
dtr.fit(X_train, y_train)
importances = dtr.feature_importances_
plt.barh(X_train.columns, importances)
plt.title("Feature Importance Plot")
plt.show()
################## Medical #####################
medical = pd.read_csv("insurance.csv")
dum_med = pd.get_dummies(medical, drop_first=True)
train, test = train_test_split(dum_med, test_size=0.3,
                               random_state=23)

X_train = train.drop('charges', axis=1)
y_train = train['charges']
X_test = test.drop('charges', axis=1)
y_test = test['charges']

lr = LinearRegression()
lr.fit(X_train, y_train)
ycap = lr.predict(X_test)
print(mean_squared_error(y_test, ycap))


errors = []
depth_values = [1,2,3,4,5,6,7,8,9,10]
for depth in depth_values:
    dtr = DecisionTreeRegressor(max_depth=depth, 
                                random_state=23)
    dtr.fit(X_train, y_train)
    ycap = dtr.predict(X_test)
    errors.append(mean_squared_error(y_test, ycap))

i_min = np.argmin(errors)
print("Best Depth =", depth_values[i_min])
print("Best Error Score =", np.min(errors))


dtr = DecisionTreeRegressor(max_depth=depth_values[i_min], 
                                random_state=23)
dtr.fit(X_train, y_train)
importances = dtr.feature_importances_
plt.barh(X_train.columns, importances)
plt.title("Feature Importance Plot")
plt.show()


#####################Housing$$$$$$$$$$$$$$
housing=pd.read_csv("Housing.csv")
dum_hous = pd.get_dummies(housing, drop_first=True)
train, test = train_test_split(dum_hous, test_size=0.3,
                               random_state=23)

X_train = train.drop('price', axis=1)
y_train = train['price']
X_test = test.drop('price', axis=1)
y_test = test['price']

lr = LinearRegression()
lr.fit(X_train, y_train)
ycap = lr.predict(X_test)
print(mean_squared_error(y_test, ycap))


errors = []
depth_values = [1,2,3,4,5,6,7,8,9,10]
for depth in depth_values:
    dtr = DecisionTreeRegressor(max_depth=depth, 
                                random_state=23)
    dtr.fit(X_train, y_train)
    ycap = dtr.predict(X_test)
    errors.append(mean_squared_error(y_test, ycap))

i_min = np.argmin(errors)
print("Best Depth =", depth_values[i_min])
print("Best Error Score =", np.min(errors))

dtr = DecisionTreeRegressor(max_depth=3, 
                                random_state=23)
dtr.fit(X_train, y_train)

importances = dtr.feature_importances_
plt.barh(X_train.columns, importances)
plt.title("Feature Importance Plot")
plt.show()


plt.figure(figsize=(15,10))
tree.plot_tree(dtr,feature_names=X_train.columns,
               filled=True,fontsize=10) 


######### Car Prices ###############
car = pd.read_csv(r"C:\Training\Kaggle\Datasets\Cars Prices\CarPrice_Assignment.csv")

car.drop(['car_ID','CarName'], axis=1, inplace=True)
dum_car = pd.get_dummies(car, drop_first=True)

train, test = train_test_split(dum_car, test_size=0.3,
                               random_state=23)

X_train = train.drop('price', axis=1)
y_train = train['price']
X_test = test.drop('price', axis=1)
y_test = test['price']

lr = LinearRegression()
lr.fit(X_train, y_train)
ycap = lr.predict(X_test)
print(mean_squared_error(y_test, ycap))


errors = []
depth_values = np.arange(1,16 )
for depth in depth_values:
    dtr = DecisionTreeRegressor(max_depth=depth, 
                                random_state=23)
    dtr.fit(X_train, y_train)
    ycap = dtr.predict(X_test)
    errors.append(mean_squared_error(y_test, ycap))

i_min = np.argmin(errors)
print("Best Depth =", depth_values[i_min])
print("Best Error Score =", np.min(errors))

dtr = DecisionTreeRegressor(max_depth=depth_values[i_min], 
                                random_state=23)
dtr.fit(X_train, y_train)

plt.figure(figsize=(40,30))
importances = dtr.feature_importances_
plt.barh(X_train.columns, importances)
plt.title("Feature Importance Plot")

plt.show()


plt.figure(figsize=(15,10))
tree.plot_tree(dtr,feature_names=X_train.columns,
               filled=True,fontsize=10) 


