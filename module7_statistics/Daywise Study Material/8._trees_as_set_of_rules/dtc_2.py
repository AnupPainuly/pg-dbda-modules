import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn import tree 
from sklearn.model_selection import train_test_split

hr = pd.read_csv("HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first=True)

X = dum_hr.drop('left', axis=1)
y = dum_hr['left']

X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,
                                                    test_size=0.3,
                                                    random_state=23)

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

