import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder 
from sklearn.naive_bayes import GaussianNB

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X = train.drop(['id','target'], axis=1)
y = train['target']

nb = GaussianNB()
nb.fit(X, y)

y_pred_prob = nb.predict_proba(test.iloc[:,1:])
submit = pd.DataFrame(y_pred_prob,
                      columns=['Class_1','Class_2','Class_3',
                               'Class_4','Class_5','Class_6',
                               'Class_7','Class_8','Class_9'])
submit['id']= test['id']

submit = submit[['id','Class_1','Class_2','Class_3',
         'Class_4','Class_5','Class_6',
         'Class_7','Class_8','Class_9']]