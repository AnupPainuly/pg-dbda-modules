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
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

brupt = pd.read_csv("Bankruptcy.csv")

X = brupt.drop(['NO','YR','D'], axis=1)
y = brupt['D']

prcomp = PCA()

scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    stratify=y,
                                                    random_state=23,
                                                    test_size=0.3)
X_scl_trn = scaler.fit_transform(X_train)
X_PC_trn = prcomp.fit_transform(X_scl_trn)
print("%age variations explained:")
print(prcomp.explained_variance_ratio_*100)

cs = np.cumsum(prcomp.explained_variance_ratio_*100)
pcs = np.arange(1,25 )
plt.scatter(pcs, cs)
plt.plot(pcs, cs)
plt.xlabel("PCs")
plt.ylabel("%age variation explained")
plt.show()

red_data_trn = X_PC_trn[:,:8]

svm = SVC(probability=True, kernel='linear', random_state=23)
svm.fit(red_data_trn, y_train)

X_scl_tst = scaler.transform(X_test)
X_PC_tst = prcomp.transform(X_scl_tst)

red_data_tst = X_PC_tst[:,:8]

y_pred = svm.predict(red_data_tst)
print(accuracy_score(y_test, y_pred))
y_pred_prob = svm.predict_proba(red_data_tst)
print(log_loss(y_test, y_pred_prob))

############# with Pipeline ######################
prcomp = PCA(n_components=8)
pipe = Pipeline([('SCL',scaler),('PCA',prcomp),('SVM',svm)])
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_prob = pipe.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

##############################################
prcomp = PCA()
scaler = StandardScaler()
svm = SVC(probability=True, kernel='linear', random_state=23)

pipe = Pipeline([('SCL',scaler),('PCA',prcomp),('SVM',svm)])
params = {'PCA__n_components':[7,8,9,10,11,12,13],
          'SVM__C':np.linspace(0.001, 10, 20)}
kfold = StratifiedKFold(n_splits=5,shuffle=True, random_state=23)
print(pipe.get_params())
gcv = GridSearchCV(pipe, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

################## Vehicle ############################
from sklearn.model_selection import RandomizedSearchCV
vehicle = pd.read_csv("Vehicle.csv")
X = vehicle.drop('Class', axis=1)
y = vehicle['Class']

prcomp = PCA()
scaler = StandardScaler()
svm = SVC(probability=True, kernel='linear', random_state=23)

pipe = Pipeline([('SCL',scaler),('PCA',prcomp),('SVM',svm)])
params = {'PCA__n_components':[9,10,11,12,13,14,15],
          'SVM__C':np.linspace(0.001, 10, 20),
          'SVM__decision_function_shape':['ovo','ovr']}
kfold = StratifiedKFold(n_splits=5,shuffle=True, random_state=23)
print(pipe.get_params())
gcv = RandomizedSearchCV(pipe, param_distributions=params,
                         verbose=2,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

