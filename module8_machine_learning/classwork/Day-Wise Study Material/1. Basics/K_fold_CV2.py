import pandas as pd
import numpy as np 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier 

brupt = pd.read_csv("Bankruptcy.csv")

X = brupt.drop(['NO','YR','D'], axis=1)
y = brupt['D']

kfold = StratifiedKFold(n_splits=5, shuffle=True, 
              random_state=23)
depth_values = [1,2,3,4,5,6,7,8,9,10]
dtc = DecisionTreeClassifier(random_state=23)
params = {'max_depth': depth_values}
print(dtc.get_params())

gcv = GridSearchCV(dtc, param_grid=params,
                   cv=kfold, 
                   scoring='neg_log_loss')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

############################################
nb = GaussianNB()
results = cross_val_score(nb, X, y,cv=kfold, 
                          scoring='neg_log_loss')
print(results.mean())