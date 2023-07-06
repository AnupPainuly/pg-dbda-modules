import pandas as pd 
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn import tree 
from sklearn.model_selection import train_test_split

telecom = pd.read_csv("Telecom.csv")

dum_tel = pd.get_dummies(telecom, drop_first=True)
dum_tel = dum_tel.astype(int)
X = dum_tel.drop('Response_Y',axis=1)
y = dum_tel['Response_Y']
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3,
                                                    random_state=23)
nb = BernoulliNB()
# Apriori Probabilities calculation
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

acc = []
depth_values = [1,2,3,4,5,6,7,8,9,10]
for depth in depth_values:
    dtc = DecisionTreeClassifier(max_depth=depth, 
                                random_state=23)
    dtc.fit(X_train, y_train)
    ycap = dtc.predict(X_test)
    acc.append(accuracy_score(y_test, ycap))

i_max = np.argmax(acc)
print("Best Depth =", depth_values[i_max])
print("Best Accuracy Score =", np.max(acc))

################## Cancer ####################
cancer = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Cancer\Cancer.csv")
dum_cancer = pd.get_dummies(cancer, drop_first=True)
X = dum_cancer.drop(['subjid','Class_recurrence-events'], axis=1)
y = dum_cancer['Class_recurrence-events']

X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,
                                                    test_size=0.3,
                                                    random_state=23)
nb = BernoulliNB()
# Apriori Probabilities calculation
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

acc = []
depth_values = [1,2,3,4,5,6,7,8,9,10]
for depth in depth_values:
    dtc = DecisionTreeClassifier(max_depth=depth, 
                                random_state=23)
    dtc.fit(X_train, y_train)
    ycap = dtc.predict(X_test)
    acc.append(accuracy_score(y_test, ycap))

i_max = np.argmax(acc)
print("Best Depth =", depth_values[i_max])
print("Best Accuracy Score =", np.max(acc))

############### Bankruptcy #####################
from sklearn.naive_bayes import GaussianNB
brupt = pd.read_csv("Bankruptcy.csv")
X = brupt.drop(['NO','YR','D'], axis=1)
y = brupt['D']

X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,
                                                    test_size=0.3,
                                                    random_state=23)
nb = GaussianNB()
# Apriori Probabilities calculation
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

nb.predict_proba(X_test)

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

##################### Glass #############
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
glass = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Glass Identification\Glass.csv")
X = glass.drop('Type', axis=1)
y = glass['Type']
lbl = LabelEncoder()
y = lbl.fit_transform(y)
print(lbl.classes_)
print(dict(zip(lbl.classes_,np.unique(y))))

X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,
                                                    test_size=0.3,
                                                    random_state=23)
nb = GaussianNB()
# Apriori Probabilities calculation
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=lbl.classes_)
disp.plot()
plt.xticks(rotation=90)
plt.show()

print(accuracy_score(y_test, y_pred))

acc = []
depth_values = [1,2,3,4,5,6,7,8,9,10]
for depth in depth_values:
    dtc = DecisionTreeClassifier(max_depth=depth, 
                                random_state=23)
    dtc.fit(X_train, y_train)
    ycap = dtc.predict(X_test)
    acc.append(accuracy_score(y_test, ycap))

i_max = np.argmax(acc)
print("Best Depth =", depth_values[i_max])
print("Best Accuracy Score =", np.max(acc))


dtc = DecisionTreeClassifier(max_depth=depth_values[i_max], 
                                random_state=23)
dtc.fit(X_train, y_train)
importances = dtc.feature_importances_
plt.barh(X_train.columns, importances)
plt.title("Feature Importance Plot")
plt.show()

y_pred = dtc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=lbl.classes_)
disp.plot()
plt.xticks(rotation=90)
plt.show()
