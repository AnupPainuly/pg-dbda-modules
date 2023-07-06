import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn import tree 
pizza = pd.read_csv("pizza.csv")
X = pizza[['Promote']]
y = pizza['Sales']

dtr = DecisionTreeRegressor(max_depth=3, random_state=23)
dtr.fit(X, y)

plt.figure(figsize=(15,10))
tree.plot_tree(dtr,feature_names=X.columns,
               filled=True,fontsize=10) 

ycap = dtr.predict(X)
print(mean_squared_error(y, ycap))

ss1y = y[X['Promote']<=54]
ss1x = X[X['Promote']<=54]

ss2y = y[X['Promote']>54] 
ss2x = X[X['Promote']>54]

print(ss1y.mean())
print(ss2y.mean())

lr = LinearRegression()
lr.fit(X, y)
ycap = lr.predict(X)
print(mean_squared_error(y, ycap))


############# Insure Auto ###########
insure = pd.read_csv("Insure_auto.csv")

X = insure[['Home','Automobile']]
y = insure['Operating_Cost'] 

dtr = DecisionTreeRegressor(max_depth=3, random_state=23)
dtr.fit(X, y)

plt.figure(figsize=(15,10))
tree.plot_tree(dtr,feature_names=X.columns,
               filled=True,fontsize=10) 

ycap = dtr.predict(X)
print(mean_squared_error(y, ycap))

lr = LinearRegression()
lr.fit(X, y)

ycap = lr.predict(X)
print(mean_squared_error(y, ycap))

############# Boston $#################
boston = pd.read_csv("Boston.csv")

X = boston.drop('medv', axis=1)
y = boston['medv']

dtr = DecisionTreeRegressor(max_depth=3, random_state=23)
dtr.fit(X, y)

plt.figure(figsize=(15,10))
tree.plot_tree(dtr,feature_names=X.columns,
               filled=True,fontsize=10) 

ycap = dtr.predict(X)
print(mean_squared_error(y, ycap))

lr = LinearRegression()
lr.fit(X, y)

ycap = lr.predict(X)
print(mean_squared_error(y, ycap))
