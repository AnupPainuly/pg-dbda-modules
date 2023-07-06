import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression

train = pd.read_csv("train.csv")
store = pd.read_csv("store.csv")
test = pd.read_csv("test.csv")

store.drop(['CompetitionDistance',
            'CompetitionOpenSinceMonth',
            'CompetitionOpenSinceYear',
            'Promo2SinceWeek',
            'Promo2SinceYear',
            'PromoInterval'], axis=1, inplace=True)
pd.isnull(store).sum()

entire_train = store.merge(train, on="Store")
entire_train['StateHoliday'] = entire_train['StateHoliday'].astype(str)
X_train = entire_train.drop(['Date','Sales','Customers'], 
                            axis=1)

pd.isnull(X_train).sum()

X_train = pd.get_dummies(X_train)
y_train = entire_train['Sales']

print(X_train.columns)

########  Test set Processing
entire_test = store.merge(test, on="Store")

open_missing = entire_test[entire_test['Open'].isnull()]
pd.isnull(entire_test).sum()

st_622_tst = entire_test[entire_test['Store']==622]
st_622_trn = entire_train[entire_train['Store']==622]

pd.crosstab(index=st_622_trn['DayOfWeek'], 
            columns=st_622_trn['Open'],
            margins=True)
pd.crosstab(index=st_622_tst['DayOfWeek'], 
            columns=st_622_tst['Open'],
            margins=True)

entire_test.loc[(entire_test['DayOfWeek']==7) & (entire_test['Store']==622),
                "Open"] = 0

entire_test.loc[(entire_test['DayOfWeek']!=7) & (entire_test['Store']==622),
                "Open"] = 1
X_test = entire_test.drop(['Date','Id'], axis=1)

pd.isnull(X_test).sum()

X_test = pd.get_dummies(X_test)

print(X_test.columns)
X_train.drop(['StateHoliday_b','StateHoliday_c'], 
             axis=1, inplace=True)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
y_pred[y_pred<0] = 0

#################### Considering Customers Column
mean_cust = entire_train.groupby(['DayOfWeek','Store'])['Customers'].mean()
mean_cust = mean_cust.reset_index()
mean_cust.rename(columns={'Customers':'mean'}, inplace=True)

sd_cust = entire_train.groupby(['DayOfWeek','Store'])['Customers'].std()
sd_cust = sd_cust.reset_index()
sd_cust.rename(columns={'Customers':'sd'}, inplace=True)

skew_cust = entire_train.groupby(['DayOfWeek','Store'])['Customers'].skew()
skew_cust = skew_cust.reset_index()
skew_cust.rename(columns={'Customers':'skew'}, inplace=True)

stats_cust = mean_cust.merge(sd_cust, 
                    on=['DayOfWeek',
                         'Store']).merge(skew_cust,
                                         on=['DayOfWeek',
                                             'Store'])

X_train = entire_train.drop(['Date','Sales','Customers'], 
                            axis=1)

X_train_cust = X_train.merge(stats_cust, on=['DayOfWeek','Store'])

X_test = entire_test.drop(['Date','Id'], axis=1)
X_test_cust = X_test.merge(stats_cust, on=['DayOfWeek','Store'])

X_train_cust = pd.get_dummies(X_train_cust)
X_test_cust = pd.get_dummies(X_test_cust)

X_train_cust.drop(['StateHoliday_b','StateHoliday_c','Store'], 
             axis=1, inplace=True)
X_test_cust.drop(['Store'], 
             axis=1, inplace=True)

lr = LinearRegression()
lr.fit(X_train_cust, y_train)

y_pred = lr.predict(X_test_cust)
y_pred[y_pred<0] = 0
