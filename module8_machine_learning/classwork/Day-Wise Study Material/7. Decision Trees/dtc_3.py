import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("BreastCancer.csv")
dum_df = pd.get_dummies(df, drop_first=True)

X = dum_df.drop(['Class_Malignant','Code'], axis=1)
y = dum_df['Class_Malignant']

clf = DecisionTreeClassifier(random_state=23)
params = {'max_depth':[None, 3, 4, 5],
          'min_samples_split':[2, 5, 10, 20],
          'min_samples_leaf': [1, 5, 10, 20]}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
gcv = GridSearchCV(clf, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

################### Satellite ####################
satellite = pd.read_csv("Satellite.csv", sep=";")
lbl = LabelEncoder()
satellite['classes'] = lbl.fit_transform(satellite['classes'])

y = satellite['classes']
X = satellite.drop('classes', axis=1)

clf = DecisionTreeClassifier(random_state=23)
params = {'max_depth':[None, 3, 4, 5, 6],
          'min_samples_split': np.arange(2,23,2),
          'min_samples_leaf': np.arange(1, 22, 2 )}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
gcv = GridSearchCV(clf, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_
plt.figure(figsize=(45,15))
tree.plot_tree(best_model,feature_names=X.columns,
               class_names=lbl.classes_,
               filled=True,fontsize=13) 

imps = best_model.feature_importances_
cols = list(X.columns)
imp_df = pd.DataFrame({'feature':cols,
                       'importance':imps})
imp_df.sort_values(by='importance', inplace=True)
plt.figure(figsize=(8,6))
plt.title("Sorted Importances Plot")
plt.barh(imp_df['feature'], imp_df['importance'])
plt.show()
