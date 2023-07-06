import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
pizza = pd.read_csv("pizza.csv")
sns.scatterplot(data=pizza, x='Promote',y='Sales')
plt.show()

X = pizza[['Promote']]
y = pizza['Sales']

lr = LinearRegression()
lr.fit(X, y)

print("b0 =", lr.intercept_)
print("b1 =", lr.coef_)

############# Insure Auto ###########
insure = pd.read_csv("Insure_auto.csv")

X = insure[['Home']]
y = insure['Operating_Cost'] 

lr = LinearRegression()
lr.fit(X, y)

print("b0 =", lr.intercept_)
print("b1 =", lr.coef_)

#ycap = insure['Home']*215.21298174+23045.63894523328
# or
ycap = lr.predict(X)
mse = np.sum((y - ycap)**2)/10
#print("mean squared error =", mse)
# OR
print(mean_squared_error(y, ycap))

X = insure[['Automobile']]
y = insure['Operating_Cost'] 

lr = LinearRegression()
lr.fit(X, y)

print("b0 =", lr.intercept_)
print("b1 =", lr.coef_)

#ycap = insure['Automobile']*111.05405476+12337.458724422118
ycap = lr.predict(X)
print(mean_squared_error(y, ycap))

X = insure[['Home','Automobile']]
y = insure['Operating_Cost'] 

lr = LinearRegression()
lr.fit(X, y)

print("b0 =", lr.intercept_)
print("b1 =", lr.coef_)

ycap = lr.predict(X)
print(mean_squared_error(y, ycap))

############# Boston $#################
boston = pd.read_csv("Boston.csv")

X = boston.drop('medv', axis=1)
y = boston['medv']

lr = LinearRegression()
lr.fit(X, y)

print("b0 =", lr.intercept_)
print("b1 =", lr.coef_)

############### Salaries ############
salaries = pd.read_csv("Salaries.csv")

dum_sals = pd.get_dummies(salaries, drop_first=True)
print(dum_sals.columns)

X = dum_sals.drop(['salary'], axis=1)
y = dum_sals['salary']

lr = LinearRegression()
lr.fit(X, y)

print("b0 =", lr.intercept_)
print("b1 =", lr.coef_)
