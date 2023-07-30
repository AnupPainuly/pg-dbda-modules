import pandas as pd 
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss 
import numpy as np 
from sklearn.pipeline import Pipeline 

brupt = pd.read_csv("Bankruptcy.csv")

X = brupt.drop(['NO','YR','D'], axis=1)
y = brupt['D']

svm = SVC(kernel='linear',probability=True,
          random_state=23)
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
results = cross_val_score(svm, X, y, 
                          scoring='neg_log_loss',
                          cv=kfold)
print(results.mean())

#scaler = StandardScaler()
scaler = MinMaxScaler()
pipe = Pipeline([('SCL',scaler),('SVM',svm)])

params = {'SVM__C':np.linspace(0.001, 5, 20)}
gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)



############# Radial #######################
params = {'C':np.linspace(0.001, 5, 20),
          'gamma':np.linspace(0.001, 5, 20)}
svm = SVC(kernel='rbf',probability=True,
          random_state=23)
gcv = GridSearchCV(svm, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

############# Polynomial #######################
params = {'C':np.linspace(0.001, 5, 20),
          'degree':[2,3,4]}
svm = SVC(kernel='poly',probability=True,
          random_state=23)
gcv = GridSearchCV(svm, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

################# Vehicle #################################
from sklearn.preprocessing import LabelEncoder
vehicle = pd.read_csv("Vehicle.csv")
lbl = LabelEncoder()
vehicle['Class'] = lbl.fit_transform(vehicle['Class'])

y = vehicle['Class']
X = vehicle.drop('Class', axis=1)

##### linear
scaler = StandardScaler()
#scaler = MinMaxScaler()
svm = SVC(kernel='linear',probability=True,
          random_state=23)
pipe = Pipeline([('SCL',scaler),('SVM',svm)])
print(pipe.get_params())
params = {'SVM__C': np.linspace(0.001,10,20),
          'SVM__decision_function_shape':['ovo','ovr']}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)

gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   scoring='neg_log_loss',
                   cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

##### radial
svm = SVC(kernel='rbf',probability=True,
          random_state=23)
pipe = Pipeline([('SCL',scaler),('SVM',svm)])
print(pipe.get_params())
params = {'SVM__C': np.linspace(0.001,10,20),
          'SVM__gamma': np.linspace(0.001,10,20),
          'SVM__decision_function_shape':['ovo','ovr']}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)

gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   scoring='neg_log_loss',
                   cv=kfold,n_jobs=-1)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

##### polynomial
svm = SVC(kernel='poly',probability=True,
          random_state=23)
pipe = Pipeline([('SCL',scaler),('SVM',svm)])
print(pipe.get_params())
params = {'SVM__C': np.linspace(0.001,10,20),
          'SVM__degree': [2,3,4],
          'SVM__decision_function_shape':['ovo','ovr']}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)

gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   scoring='neg_log_loss',
                   cv=kfold,n_jobs=-1)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
