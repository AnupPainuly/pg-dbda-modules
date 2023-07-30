import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder 
from sklearn.naive_bayes import GaussianNB

wisconsin = pd.read_csv("BreastCancer.csv")
X = wisconsin.drop(['Code', 'Class'], axis=1)
y = wisconsin['Class']

le = LabelEncoder()
y = le.fit_transform(y)
dict(zip(le.classes_,np.unique(y)))

nb = GaussianNB()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
results = cross_val_score(nb, X, y, scoring='neg_log_loss',
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

############## Glass Identification ################
glass = pd.read_csv("Glass.csv")
le = LabelEncoder()

X = glass.drop('Type', axis=1)
y = glass['Type']
y = le.fit_transform(y)
dict(zip(le.classes_,np.unique(y)))

params = {'penalty':['l1','l2','elasticnet',None],
          'multi_class':['ovr', 'multinomial'],
          'C': np.linspace(0.001, 5, 10),
          'l1_ratio':np.linspace(0,1,10)}
lr = LogisticRegression()
gcv = GridSearchCV(lr, param_grid=params,verbose=3,
                   scoring='neg_log_loss',
                   cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_
################ Predict on the unlabelled data #########
tst_glass = pd.read_csv("tst_Glass.csv")
y_pred = best_model.predict(tst_glass)

print(le.inverse_transform(y_pred))








