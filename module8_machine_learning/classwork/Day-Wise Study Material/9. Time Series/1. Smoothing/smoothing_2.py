import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error as mse
import numpy as np 
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from ISLP import load_data 

NYSE = load_data('NYSE')
print(NYSE.columns)

NYSE['DJ_return'].plot()
plt.plot()

y = NYSE['DJ_return']
y_train = y[NYSE['train']==True]
y_test = y[NYSE['train']==False]


################################################
Smarket = load_data('Smarket')
print(Smarket.columns)

Smarket['Volume'].plot()
plt.show()

Smarket.index = pd.date_range(start='1/1/2001',
              end='10/15/2005',
              freq="B")

