import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn import tree 
from sklearn.model_selection import train_test_split
brupt = pd.read_csv("Bankruptcy.csv")
train, test = train_test_split(brupt, test_size=0.3,
                               random_state=23)

X_train = train.drop(['NO','YR','D'], axis=1)
y_train = train['D']
X_test = test.drop(['NO','YR','D'], axis=1)
y_test = test['D']

dtc = DecisionTreeClassifier(max_depth=3, random_state=23)
dtc.fit(X_train, y_train)

plt.figure(figsize=(15,10))
tree.plot_tree(dtc,feature_names=X_train.columns,
               filled=True,fontsize=15)
y_pred = dtc.predict(X_test)

# comp = pd.DataFrame({'Existing':y_test,
#                      'Predicted':y_pred})
# pd.crosstab(index=comp['Existing'], 
#             columns=comp['Predicted'])
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

dtc = DecisionTreeClassifier(max_depth=depth_values[i_max], 
                                random_state=23)
dtc.fit(X_train, y_train)
importances = dtc.feature_importances_
plt.barh(X_train.columns, importances)
plt.title("Feature Importance Plot")
plt.show()

############### Bank #################
bank = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\bank\bank.csv",
                   sep=";")
dum_bank = pd.get_dummies(bank, drop_first=True)

train, test = train_test_split(dum_bank, test_size=0.3,
                               random_state=23)

X_train = train.drop(['y_yes'], axis=1)
y_train = train['y_yes']
X_test = test.drop(['y_yes'], axis=1)
y_test = test['y_yes']

# comp = pd.DataFrame({'Existing':y_test,
#                      'Predicted':y_pred})
# pd.crosstab(index=comp['Existing'], 
#             columns=comp['Predicted'])
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
plt.figure(figsize=(15,10))
importances = dtc.feature_importances_
plt.barh(X_train.columns, importances)
plt.title("Feature Importance Plot")
plt.show()
