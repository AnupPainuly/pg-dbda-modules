import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import StackingClassifier 
from sklearn.ensemble import RandomForestClassifier

cancer = pd.read_csv("BreastCancer.csv", index_col=0)
cancer.head()

lbl = LabelEncoder()
cancer['Class'] = lbl.fit_transform(cancer['Class'])
X = cancer.drop('Class', axis=1)
y = cancer['Class']

X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,
                                                 random_state=23,
                                                 test_size=0.3)
lr = LogisticRegression()
svm = SVC(probability=True, random_state=23)
dtc = DecisionTreeClassifier(random_state=23)
models = [('LR', lr),('SVM',svm ),('TREE',dtc)]
gbm = XGBClassifier(random_state=23)
stack = StackingClassifier(estimators=models,
                           final_estimator=gbm,
                           passthrough=True)
stack.fit(X_train, y_train)

y_pred_prob = stack.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob[:,1]))

############# final = rf ##################
rf = RandomForestClassifier(random_state=23)
stack = StackingClassifier(estimators=models,
                           final_estimator=rf,
                           passthrough=True)
stack.fit(X_train, y_train)

y_pred_prob = stack.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob[:,1]))

####### Grid Search
print(stack.get_params())
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
params = {'LR__penalty':['l1','l2','elasticnet',None],
          'SVM__C':np.linspace(0.001, 5, 5),
          'SVM__gamma':np.linspace(0.001, 5, 5),
          'TREE__max_depth':[None, 2,4],
          'TREE__min_samples_split':[2, 5, 10],
          'final_estimator':[rf, gbm],
          'final_estimator__n_estimators':[50,100, 150]}
rgcv = RandomizedSearchCV(stack, n_iter=20, n_jobs=-1,
                          param_distributions=params,
                   cv=kfold, scoring='neg_log_loss')
rgcv.fit(X, y)
print(rgcv.best_params_)
print(rgcv.best_score_)
