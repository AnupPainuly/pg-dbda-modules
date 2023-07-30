import pandas as pd 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss 
import numpy as np 
brupt = pd.read_csv("Bankruptcy.csv")

X = brupt.drop(['NO','YR','D'], axis=1)
y = brupt['D']

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3,
                               stratify=y,
                               random_state=23)

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_prob = nb.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob[:,1]))

scores = []
depth_values = [1,2,3,4,5,6,7,8,9,10]
for depth in depth_values:
    dtc = DecisionTreeClassifier(max_depth=depth, 
                                random_state=23)
    dtc.fit(X_train, y_train)
    y_pred_prob = dtc.predict_proba(X_test)
    scores.append(log_loss(y_test, y_pred_prob[:,1]))

i_min = np.argmin(scores)
print("Best Depth =", depth_values[i_min])
print("Best Log Loss Score =", np.min(scores))