import pandas as pd 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis 
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

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred_prob = lda.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob[:,1]))

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
y_pred_prob = qda.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob[:,1]))