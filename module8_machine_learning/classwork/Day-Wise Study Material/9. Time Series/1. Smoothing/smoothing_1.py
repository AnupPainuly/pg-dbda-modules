import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error as mse
import numpy as np 
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing, Holt

df = pd.read_csv("monthly-milk-production-pounds-p.csv")
df.head()

df.plot()
plt.show()

series = df['Milk']
result = seasonal_decompose(series, model='additive',period=12)
result.plot()
plt.show()

result = seasonal_decompose(series, model='multiplicative',period=12)
result.plot()
plt.show()

y = df['Milk']
#### Centered MA
fcast = y.rolling(3,center=True).mean()
plt.plot(y, label='Data',color='blue')
plt.plot(fcast, label='Moving Average Forecast',
         color='red')
plt.legend(loc='best')
plt.show()

y_train = df['Milk'][:-12]
y_test = df['Milk'][-12:]
plt.plot(y_train, label='Train',color='blue')
plt.plot(y_test, label='Test',color='orange')
plt.legend(loc='best')
plt.show()


#### Trailing Rolling Mean
fcast = y_train.rolling(5,center=False).mean()
lastMA = fcast.iloc[-1]
fSeries = pd.Series(lastMA.repeat(len(y_test)))
MA_fcast = pd.concat([fcast, fSeries],
                     ignore_index=True)
plt.plot(y_train, label='Train',color='blue')
plt.plot(MA_fcast, label='Moving Average Forecast',
         color='red')
plt.plot(y_test, label="Test")
plt.legend(loc='best')
plt.show()

print("MSE =",mse(y_test, fSeries))

alpha = 0.1
# Simple Exponential Smoothing
fit1 = SimpleExpSmoothing(y_train).fit(smoothing_level=alpha)
fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.legend(loc='best')
plt.show()

print("MSE =",mse(y_test, fcast1))


# Holt's Linear Method
alpha = 0.8
beta = 0.02
### Linear Trend
fit1 = Holt(y_train).fit(smoothing_level=alpha, 
                         smoothing_trend=beta)
fcast1 = fit1.forecast(len(y_test))

# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.text(100,600,"MSE="+str(round(mse(y_test, fcast1),2)))
plt.title("Holt's Linear Trend")
plt.legend(loc='best')
plt.show()

print("MSE =",mse(y_test, fcast1))



# Holt's Exponential Method
alpha = 0.8
beta = 0.02
### Linear Trend
fit1 = Holt(y_train, exponential=True).fit(smoothing_level=alpha, 
                         smoothing_trend=beta)
fcast1 = fit1.forecast(len(y_test))

# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.text(100,600,"MSE="+str(round(mse(y_test, fcast1),2)))
plt.title("Holt's Exponential Trend")
plt.legend(loc='best')
plt.show()

print("MSE =",mse(y_test, fcast1))


### Additive Damped Trend
alpha = 0.8
beta = 0.02
phi = 0.4
fit3 = Holt(y_train, damped_trend=True).fit(smoothing_level=alpha,
                                      damping_trend= phi,
                                      smoothing_trend=beta)
fcast3 = fit3.forecast(len(y_test))

# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast3.plot(color="purple", label='Forecast')
plt.text(100,600,"MSE="+str(round(mse(y_test, fcast1),2)))
plt.title("Holt's Additive Damped Trend")
plt.legend(loc='best')
plt.show()

### Multiplicative Damped Trend
alpha = 0.8
beta = 0.02
phi = 0.4
fit3 = Holt(y_train, damped_trend=True,
            exponential=True).fit(smoothing_level=alpha,
                                      damping_trend= phi,
                                      smoothing_trend=beta)
fcast3 = fit3.forecast(len(y_test))

# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast3.plot(color="purple", label='Forecast')
plt.text(100,600,"MSE="+str(round(mse(y_test, fcast3),2)))
plt.title("Holt's Multiplicative Damped Trend")
plt.legend(loc='best')
plt.show()

# Holt-Winters' Method

########### Additive #####################
alpha = 0.8
beta = 0.02
phi = 0.4
gamma = 0.3
fit1 = ExponentialSmoothing(y_train, 
                            seasonal_periods=12, 
                            trend='add', 
                            seasonal='add').fit(smoothing_level=alpha,
                                                smoothing_trend=beta,
                                                smoothing_seasonal=gamma)

fcast1 = fit1.forecast(len(y_test)).rename("Holt-Winters Additive Trend and seasonality")
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.text(100,600,"MSE="+str(round(mse(y_test, fcast1),2)))
plt.legend(loc='best')
plt.show()

### auto-tuning
fit1 = ExponentialSmoothing(y_train, 
                            seasonal_periods=12, 
                            trend='add', 
                            seasonal='add').fit()

fcast1 = fit1.forecast(len(y_test)).rename("Holt-Winters Additive Trend and seasonality")
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.text(100,600,"MSE="+str(round(mse(y_test, fcast1),2)))
plt.legend(loc='best')
plt.show()


########### Mult #####################
alpha = 0.8
beta = 0.02
phi = 0.4
gamma = 0.3
fit1 = ExponentialSmoothing(y_train, 
                            seasonal_periods=12, 
                            trend='add', 
                            seasonal='mul').fit(smoothing_level=alpha,
                                                smoothing_trend=beta,
                                                smoothing_seasonal=gamma)

fcast1 = fit1.forecast(len(y_test)).rename("Holt-Winters Additive Trend and seasonality")
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.text(100,600,"MSE="+str(round(mse(y_test, fcast1),2)))
plt.legend(loc='best')
plt.show()

## Auto-Tune
fit1 = ExponentialSmoothing(y_train, 
                            seasonal_periods=12, 
                            trend='add', 
                            seasonal='mul').fit()

fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.text(100,600,"MSE="+str(round(mse(y_test, fcast1),2)))
plt.legend(loc='best')
plt.show()

### HW Add with Damping #################

alpha = 0.8
beta = 0.02
phi = 0.4
gamma = 0.3
fit1 = ExponentialSmoothing(y_train, 
                            seasonal_periods=12, 
                            trend='add', 
                            damped_trend=True,
                            seasonal='add').fit(smoothing_level=alpha,
                                                smoothing_trend=beta,
                                                smoothing_seasonal=gamma,
                                                damping_trend=phi)

fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.text(100,600,"MSE="+str(round(mse(y_test, fcast1),2)))
plt.legend(loc='best')
plt.show()

### auto-tuning
fit1 = ExponentialSmoothing(y_train, 
                            seasonal_periods=12, 
                            trend='add', 
                            damped_trend=True,
                            seasonal='add').fit()

fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.text(100,600,"MSE="+str(round(mse(y_test, fcast1),2)))
plt.legend(loc='best')
plt.show()

###### HW multi with damping
alpha = 0.8
beta = 0.02
phi = 0.4
gamma = 0.3
fit1 = ExponentialSmoothing(y_train, 
                            seasonal_periods=12, 
                            trend='add', 
                            damped_trend=True,
                            seasonal='mul').fit(smoothing_level=alpha,
                                                smoothing_trend=beta,
                                                smoothing_seasonal=gamma,
                                                damping_trend=phi)

fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.text(100,600,"MSE="+str(round(mse(y_test, fcast1),2)))
plt.legend(loc='best')
plt.show()

### auto-tuning
fit1 = ExponentialSmoothing(y_train, 
                            seasonal_periods=12, 
                            trend='add', 
                            damped_trend=True,
                            seasonal='mul').fit()

fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.text(100,600,"MSE="+str(round(mse(y_test, fcast1),2)))
plt.legend(loc='best')
plt.show()


