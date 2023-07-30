import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from ISLP import load_data

BrainCancer = load_data('BrainCancer')
BrainCancer.columns

dum_cancer = pd.get_dummies(BrainCancer, drop_first=True)
X = dum_cancer.drop(['status', 'time'], axis=1)
y = dum_cancer['status']

kfold = StratifiedKFold(n_splits=5, 
                        shuffle=True, random_state=23)

#### NB
nb = GaussianNB()
results = cross_val_score(nb, X, y,
                          scoring='neg_log_loss',
                          cv=kfold)
print(results.mean())
eval_df = pd.DataFrame({'Model':["Gaussian NB"],
                        'Params':[''],
                        'Score':[results.mean()]})
#### K-NN
### Standard Scaler
scaler = StandardScaler()
knn = KNeighborsClassifier()
params = {'KNN__n_neighbors': np.arange(1,16)}
pipe = Pipeline([('SCL',scaler),('KNN',knn)])
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
knn = KNeighborsClassifier()
gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   scoring='neg_log_loss',
                   cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
eval_df.loc[len(eval_df.index)] = ['K-NN',gcv.best_params_,
                                   gcv.best_score_]

#### Logistic
lr = LogisticRegression()
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
eval_df.loc[len(eval_df.index)] = ['Logistic',gcv.best_params_,
                                   gcv.best_score_]


##### Binning the numeric
dum_cancer['bin_gtv'] = pd.cut(dum_cancer['gtv'], 6)
dum_cancer['bin_ki'] = pd.cut(dum_cancer['ki'], 4)
dum_cancer = pd.get_dummies(dum_cancer, drop_first=True)
X = dum_cancer.drop(['status', 'time', 'gtv', 'ki'], axis=1)
y = dum_cancer['status']

nb = BernoulliNB()
results = cross_val_score(nb, X, y,
                          scoring='neg_log_loss',
                          cv=kfold)
print(results.mean())
eval_df.loc[len(eval_df.index)] = ['Bernoulli NB',"",
                                   results.mean()]
eval_df.sort_values("Score", ascending=False)
