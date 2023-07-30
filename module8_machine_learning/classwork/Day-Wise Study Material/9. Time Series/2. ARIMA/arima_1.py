import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
import numpy as np 
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv("monthly-milk-production-pounds-p.csv")
df.head()

## Augmented Dicky-Fuller Test
def adfuller_test(ts):
    adfuller_result = adfuller(ts, autolag=None)
    adfuller_out = pd.Series(adfuller_result[0:4],
    index=[ 'Test Statistic', 'p-value', 'Lags Used',
    'Number of Observations Used'])
    print(adfuller_out)
    
adfuller_test(df['Milk'])
diff_milk = df['Milk'].diff()
adfuller_test(diff_milk.dropna())

##### Autocorrelation ######
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df['Milk'], lags=5)
plt.show()

coal = pd.read_csv("Coal Consumption.csv")
coal.head()
adfuller_test(coal['Amount'])
diff_amt = coal['Amount'].diff()
adfuller_test(diff_amt.dropna())

plot_acf(coal['Amount'], lags=5)
plt.show()


####### Auto Reg ##############
from statsmodels.tsa.ar_model import AutoReg
y = df['Milk']
y_train = df['Milk'][:-12]
y_test = df['Milk'][-12:]
model = AutoReg(y_train, lags=3)
model_fit = model.fit()

print('Coefficients: %s' % model_fit.params)
predictions = model_fit.predict(start=len(y_train), 
                                end=len(y_train)+len(y_test)-1, 
                                dynamic=False)

print("MSE =",mse(y_test, predictions))

################# Moving Average #########################
from statsmodels.tsa.arima.model import ARIMA
# train MA
model = ARIMA(y_train,order=(0,0,2))
model_fit = model.fit()

print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(y_train), 
                                end=len(y_train)+len(y_test)-1, 
                                dynamic=False)
print("MSE =",mse(y_test, predictions)) 

############### ARMA ######################################
# 1st order AR, 2nd order MA
model = ARIMA(y_train,order=(1,0,2))
model_fit = model.fit()

print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(y_train), 
                                end=len(y_train)+len(y_test)-1, 
                                dynamic=False)
print("MSE =",mse(y_test, predictions)) 


############### ARIMA ######################################
# 1st order AR, 1st order Diff, 2nd order MA
model = ARIMA(y_train,order=(1,1,2))
model_fit = model.fit()

print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(y_train), 
                                end=len(y_train)+len(y_test)-1, 
                                dynamic=False)
print("MSE =",mse(y_test, predictions)) 

########### pmdarima ##############################

from pmdarima.arima import auto_arima

########## ARIMA ####################
model = auto_arima(y_train, trace=True,
                   error_action='ignore', 
                   suppress_warnings=True)
forecast = model.predict(n_periods=len(y_test))
forecast = pd.DataFrame(forecast,index = y_test.index,
                        columns=['Prediction'])
print("MSE =",mse(y_test, forecast)) 

#plot the predictions for validation set
plt.plot(y_train, label='Train',color="blue")
plt.plot(y_test, label='Valid',color="pink")
plt.plot(forecast, label='Prediction',color="purple")
plt.legend(loc='best')
plt.show()

############# SARIMA ################
model = auto_arima(y_train, trace=True,seasonal=True,m=12,
                   error_action='ignore', 
                   suppress_warnings=True)
forecast = model.predict(n_periods=len(y_test))
forecast = pd.DataFrame(forecast,index = y_test.index,
                        columns=['Prediction'])
print("MSE =",mse(y_test, forecast)) 

#plot the predictions for validation set
plt.plot(y_train, label='Train',color="blue")
plt.plot(y_test, label='Valid',color="pink")
plt.plot(forecast, label='Prediction',color="purple")
plt.legend(loc='best')
plt.show()

