import pandas as pd 
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss 
import numpy as np 
from sklearn.pipeline import Pipeline 

satellite = pd.read_csv("Satellite.csv", sep=";")
lbl = LabelEncoder()
satellite['classes'] = lbl.fit_transform(satellite['classes'])

y = satellite['classes']
X = satellite.drop('classes', axis=1)

svm = SVC(kernel='rbf',probability=True,
          random_state=23)
scaler = MinMaxScaler()
pipe = Pipeline([('SCL',scaler),('SVM',svm)])
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
params = {'SVM__C':np.linspace(0.001, 10, 10),
          'SVM__gamma': np.linspace(0.001, 10, 10),
          'SVM__decision_function_shape':['ovo','ovr']}
rgcv = RandomizedSearchCV(pipe, param_distributions=params,
                          verbose=3,n_jobs=-1,
                          random_state=23,
                   cv=kfold, scoring='neg_log_loss')
rgcv.fit(X, y)
print(rgcv.best_params_)
print(rgcv.best_score_)

best_model = rgcv.best_estimator_



######### Predicting on Unlabelled Data ############
tst_sat = pd.read_csv("tst_satellite.csv")
y_pred = best_model.predict(tst_sat)
print(lbl.inverse_transform(y_pred))

################### Image Segmentation ####################
glass = pd.read_csv("Glass.csv")
