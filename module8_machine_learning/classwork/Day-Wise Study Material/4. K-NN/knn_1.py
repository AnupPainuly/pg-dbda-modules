import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder 


wisconsin = pd.read_csv("BreastCancer.csv")
X = wisconsin.drop(['Code', 'Class'], axis=1)
y = wisconsin['Class']

le = LabelEncoder()
y = le.fit_transform(y)
dict(zip(le.classes_,np.unique(y)))

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    stratify=y,
                                                    test_size=0.3,
                                                    random_state=23)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_prob = knn.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

#################### Grid Search CV ####################
params = {'n_neighbors': np.arange(1,27,2)}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
knn = KNeighborsClassifier()
gcv = GridSearchCV(knn, param_grid=params,verbose=3,
                   scoring='neg_log_loss',
                   cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

################# Vehicle #################################
from sklearn.preprocessing import StandardScaler
vehicle = pd.read_csv("Vehicle.csv")
lbl = LabelEncoder()
vehicle['Class'] = lbl.fit_transform(vehicle['Class'])

y = vehicle['Class']
X = vehicle.drop('Class', axis=1)

scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    stratify=y,
                                                    test_size=0.3,
                                                    random_state=23)

X_scl_trn = scaler.fit_transform(X_train)
knn = KNeighborsClassifier(n_neighbors=25)
knn.fit(X_scl_trn, y_train)
X_scl_tst = scaler.transform(X_test)
y_pred_prob = knn.predict_proba(X_scl_tst)
print(log_loss(y_test, y_pred_prob))

######### with pipeline
from sklearn.pipeline import Pipeline
pipe = Pipeline([('SCL',scaler),('KNN',knn)])
knn = KNeighborsClassifier(n_neighbors=25)
pipe.fit(X_train, y_train)
y_pred_prob = pipe.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))
##### Grid Search
print(pipe.get_params())
params = {'KNN__n_neighbors': np.arange(1,27,2)}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
knn = KNeighborsClassifier()
gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   scoring='neg_log_loss',
                   cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

##### Min Max Scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
##### Grid Search
pipe = Pipeline([('SCL',scaler),('KNN',knn)])
print(pipe.get_params())
params = {'KNN__n_neighbors': np.arange(1,27,2)}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
knn = KNeighborsClassifier()
gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   scoring='neg_log_loss',
                   cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

############### Glass Identification ###################
glass = pd.read_csv("Glass.csv")
le = LabelEncoder()

X = glass.drop('Type', axis=1)
y = glass['Type']
y = le.fit_transform(y)
# dict(zip(le.classes_,np.unique(y)))
### Standard Scaler
scaler = StandardScaler()
knn = KNeighborsClassifier()
params = {'KNN__n_neighbors': np.arange(1,31)}
pipe = Pipeline([('SCL',scaler),('KNN',knn)])
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
knn = KNeighborsClassifier()
gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   scoring='neg_log_loss',
                   cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
### MinMax Scaler
scaler = MinMaxScaler()
knn = KNeighborsClassifier()
params = {'KNN__n_neighbors': np.arange(1,31)}
pipe = Pipeline([('SCL',scaler),('KNN',knn)])
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
knn = KNeighborsClassifier()
gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   scoring='neg_log_loss',
                   cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
