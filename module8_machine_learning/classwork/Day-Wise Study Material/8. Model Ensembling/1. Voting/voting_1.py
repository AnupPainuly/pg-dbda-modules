import pandas as pd 
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss 
import numpy as np 
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import VotingClassifier

brupt = pd.read_csv("Bankruptcy.csv")

X = brupt.drop(['NO','YR','D'], axis=1)
y = brupt['D']
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    stratify=y,
                                                    test_size=0.3,
                                                    random_state=23)
nb = GaussianNB()
lr = LogisticRegression()
lda = LinearDiscriminantAnalysis()
voting = VotingClassifier([('NB',nb),
                           ('LR',lr),
                           ('LDA',lda)], voting='soft')
voting.fit(X_train, y_train)
#y_pred = voting.predict(X_test)
y_pred_prob = voting.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob[:,1]))

############### K-fold CV #############################
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
results = cross_val_score(voting, X, y, cv=kfold,
                          scoring='neg_log_loss')
print(results.mean())

#################### Grid Search CV ################
print(voting.get_params()) 


params = {'LR__penalty':['l1','l2','elasticnet',None],
          'LR__C': np.linspace(0.001, 5, 10),
          'LR__l1_ratio':np.linspace(0,1,10)}
gcv = GridSearchCV(voting, param_grid=params,verbose=3,
                   scoring='neg_log_loss',
                   cv=kfold)
gcv.fit(X, y)

df_results = pd.DataFrame(gcv.cv_results_)
df_results.to_csv("GS_results.csv", index=False)
print(gcv.best_params_)
print(gcv.best_score_)


##############################################
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 

dtc = DecisionTreeClassifier(random_state=23)
svm = SVC(probability=True, random_state=23)
scaler = StandardScaler() 
pipe_svm = Pipeline([('SCL', scaler),('SVC', svm)])
voting = VotingClassifier([('TREE',dtc),
                           ('LR',lr),
                          ('SVM',pipe_svm)], voting='soft')
print(voting.get_params())
params = {'LR__penalty':['l1','l2','elasticnet',None],
          'LR__C': np.linspace(0.001, 5, 10),
          'LR__l1_ratio':np.linspace(0,1,10),
          'TREE__max_depth':[None, 3, 4, 5],
          'TREE__min_samples_split':[2, 5, 10, 20],
          'TREE__min_samples_leaf': [1, 5, 10, 20],
          'SVM__SVC__C':np.linspace(0.001, 10, 10),
         'SVM__SVC__gamma': np.linspace(0.001, 10, 10)}
rgcv = RandomizedSearchCV(voting, 
                          param_distributions=params,
                          n_iter=100,
                          random_state=23,
                          verbose=3,
                          scoring='neg_log_loss',
                          cv=kfold)
rgcv.fit(X,y)
print(rgcv.best_params_)
print(rgcv.best_score_)
