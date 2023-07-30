import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

hr = pd.read_csv("HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first=True)

X = dum_hr.drop('left', axis=1)
y = dum_hr['left']

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    stratify=y,
                                                    test_size=0.3,
                                                    random_state=23)

lr = LogisticRegression()
lr.fit(X_train, y_train)
print(lr.coef_)
y_pred_prob = lr.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob[:,1]))

############# K-Fold CV ###############################
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
results = cross_val_score(lr, X, y,scoring='neg_log_loss',
                          cv=kfold)
print(results.mean())

############### Grid Search CV ##########################
params = {'penalty':['l1','l2','elasticnet',None],
          'C': np.linspace(0.001, 5, 10),
          'l1_ratio':np.linspace(0,1,10)}
lr = LogisticRegression()
gcv = GridSearchCV(lr, param_grid=params,verbose=3,
                   scoring='neg_log_loss',
                   cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)



