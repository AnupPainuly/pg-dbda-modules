import pandas as pd 
#install by using pip install scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
pizza = pd.read_csv("/home/darkstar/Documents/pg-dbda/module7_statistics/Daywise Study Material/Datasets/pizza.csv")
sns.scatterplot(data=pizza, y='Promote', x='Sales')
sns.heatmap(pizza.corr(), annot=True )

# x is the independent variable, predictor
# also x is pandas Dataframe
x = pizza[['Promote']]

# y is the response variable, dependent variable, the one you want to predict
# y is pandas series
y = pizza['Sales']

lr = LinearRegression()
lr.fit(x, y)

print("β0 =", lr.intercept_)
print("β1 =", lr.coef_)
