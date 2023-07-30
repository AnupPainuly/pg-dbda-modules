import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder 
from sklearn.naive_bayes import GaussianNB

vehicle = pd.read_csv("Vehicle.csv")
lbl = LabelEncoder()
vehicle['Class'] = lbl.fit_transform(vehicle['Class'])

y = vehicle['Class']
X = vehicle.drop('Class', axis=1)

nb = GaussianNB()
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
results = cross_val_score(nb, X, y, 
                          scoring='neg_log_loss',
                          cv=kfold)
print(results.mean())

############# Grid Search Cv #################
lr = LogisticRegression()
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
