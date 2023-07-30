import pandas as pd 
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss 
import numpy as np 
from sklearn.pipeline import Pipeline 

brupt = pd.read_csv("Bankruptcy.csv")

X = brupt.drop(['NO','YR','D'], axis=1)
y = brupt['D']

############# Tree
clf = DecisionTreeClassifier(random_state=23)
params = {'max_depth':[None, 3, 4, 5],
          'min_samples_split':[2, 5, 10, 20],
          'min_samples_leaf': [1, 5, 10, 20]}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
gcv_tree = GridSearchCV(clf, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv_tree.fit(X, y)
print(gcv_tree.best_params_)
print(gcv_tree.best_score_)

bm_tree = gcv_tree.best_estimator_

imps = bm_tree.feature_importances_
cols = list(X.columns)
imp_df = pd.DataFrame({'feature':cols,
                       'importance':imps})
imp_df.sort_values(by='importance', inplace=True)
plt.figure(figsize=(8,6))
plt.title("Sorted Importances Plot for Single Tree")
plt.barh(imp_df['feature'], imp_df['importance'])
plt.show()

############### RF
rf = RandomForestClassifier(random_state=23)
params = {'max_features':[2,3,4,5,6,7]}
gcv_rf = GridSearchCV(rf, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv_rf.fit(X, y)
print(gcv_rf.best_params_)
print(gcv_rf.best_score_)


bm_rf = gcv_rf.best_estimator_

imps = bm_rf.feature_importances_
cols = list(X.columns)
imp_df = pd.DataFrame({'feature':cols,
                       'importance':imps})
imp_df.sort_values(by='importance', inplace=True)
plt.figure(figsize=(8,6))
plt.title("Sorted Importances Plot for Random Forest")
plt.barh(imp_df['feature'], imp_df['importance'])
plt.show()

################## HR #############################
hr = pd.read_csv("HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first=True)

X = dum_hr.drop('left', axis=1)
y = dum_hr['left']
