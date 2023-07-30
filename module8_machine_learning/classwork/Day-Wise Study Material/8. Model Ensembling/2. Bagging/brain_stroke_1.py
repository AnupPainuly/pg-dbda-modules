import pandas as pd 
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import MinMaxScaler 
import numpy as np 
from sklearn.metrics import log_loss, accuracy_score 

brain = pd.read_csv(r"C:\Training\Kaggle\Datasets\Brain Stroke\brain_stroke.csv")
dum_brain = pd.get_dummies(brain, drop_first=True)
print(dum_brain.columns)

y = dum_brain['stroke']
X = dum_brain.drop('stroke', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    stratify=y,
                                                    test_size=0.15,
                                                    random_state=23)

scaler = MinMaxScaler()
svm = SVC(kernel='rbf', probability=True, random_state=23)
pipe_svm = Pipeline([('SCL',scaler),('SVM',svm)])
print(pipe_svm.get_params())                                                
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
params = {'SVM__C':np.linspace(0.001, 10, 10),
          'SVM__gamma': np.linspace(0.001, 10, 10)}
rgcv_svm = RandomizedSearchCV(pipe_svm, param_distributions=params,
                          verbose=3,n_jobs=-1,
                          random_state=23,
                   cv=kfold, scoring='neg_log_loss')
rgcv_svm.fit(X_train, y_train)
print(rgcv_svm.best_params_)
print(rgcv_svm.best_score_)

best_svm = rgcv_svm.best_estimator_ 
y_pred_svm = best_svm.predict(X_test)
print("Accuracy with SVM =",accuracy_score(y_test, y_pred_svm))
y_prob_svm = best_svm.predict_proba(X_test)
print("Log Loss with SVM =",log_loss(y_test, y_prob_svm[:,1]))

################## Logistic ###########################

params = {'penalty':['l1','l2','elasticnet',None],
          'C': np.linspace(0.001, 5, 10),
          'l1_ratio':np.linspace(0,1,10)}
lr = LogisticRegression()
gcv_lr = GridSearchCV(lr, param_grid=params,verbose=3,
                   scoring='neg_log_loss',
                   cv=kfold)
gcv_lr.fit(X_train, y_train)
print(gcv_lr.best_params_)
print(gcv_lr.best_score_)

best_lr = gcv_lr.best_estimator_ 
y_pred_lr = best_lr.predict(X_test)
print("Accuracy with LR =",accuracy_score(y_test, y_pred_lr))
y_prob_lr = best_lr.predict_proba(X_test)
print("Log Loss with LR =",log_loss(y_test, y_prob_lr[:,1]))

################### Decision Tree #######################
clf = DecisionTreeClassifier(random_state=23)
params = {'max_depth':[None, 3, 4, 5],
          'min_samples_split':[2, 5, 10, 20,30, 50],
          'min_samples_leaf': [1, 5, 10, 20, 30, 50]}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
gcv_tree = GridSearchCV(clf, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv_tree.fit(X_train, y_train)
print(gcv_tree.best_params_)
print(gcv_tree.best_score_)


best_tree = gcv_tree.best_estimator_ 
y_pred_tree = best_tree.predict(X_test)
print("Accuracy with Tree =",accuracy_score(y_test, y_pred_tree))
y_prob_tree = best_tree.predict_proba(X_test)
print("Log Loss with Tree =",log_loss(y_test, y_prob_tree[:,1]))

############### Bagging with best model ###################
from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(estimator=best_lr,random_state=23)
bagging.fit(X_train, y_train)
y_pred_bag = bagging.predict(X_test)
print("Accuracy with Bag =",accuracy_score(y_test, y_pred_bag))
y_prob_bag = bagging.predict_proba(X_test)
print("Log Loss with Bag =",log_loss(y_test, y_prob_bag[:,1]))

#### Grid Search 
params = {'n_estimators':[10, 25, 50]}
gcv_bag_lr = GridSearchCV(bagging, param_grid=params,n_jobs=-1,
                   cv=kfold, scoring='neg_log_loss')
gcv_bag_lr.fit(X_train, y_train)
print(gcv_bag_lr.best_params_)
print(gcv_bag_lr.best_score_)

best_bag_lr = gcv_bag_lr.best_estimator_
y_pred_bag = best_bag_lr.predict(X_test)
print("Accuracy with Bag with LR =",accuracy_score(y_test, y_pred_bag))
y_prob_bag = best_bag_lr.predict_proba(X_test)
print("Log Loss with Bag with LR  =",log_loss(y_test, y_prob_bag[:,1]))

