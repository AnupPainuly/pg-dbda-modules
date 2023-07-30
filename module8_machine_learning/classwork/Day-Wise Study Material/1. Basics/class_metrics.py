import pandas as pd 
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

################## Cancer ####################
cancer = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Cancer\Cancer.csv")
dum_cancer = pd.get_dummies(cancer, drop_first=True)
dum_cancer = dum_cancer.astype(int)
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

print(classification_report(y_test, y_pred))

######## Tree

dtc = DecisionTreeClassifier(random_state=23,
                             max_depth=3)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)


plt.figure(figsize=(20,10))
tree.plot_tree(dtc,feature_names=X_train.columns,
               filled=True,fontsize=15,
               class_names=['0','1'])

y_pred_prob = dtc.predict_proba(X_test)

comp_probs1 = pd.DataFrame(y_pred_prob, 
                           columns=['P(y=0)',
                                    'P(y=1)'])
comp_probs1['Actual'] = y_test.values 
comp_probs1['Predicted'] = y_pred                           
comp_probs1.to_csv(r"C:\Training\Academy\Statistics (Python)\Demo Codes DBDA Jun 2023\comp_prob1.csv")

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import log_loss
y_pred_prob = dtc.predict_proba(X_test)
roc_results = roc_curve(y_test, y_pred_prob[:,1])
fpr = roc_results[0]
tpr = roc_results[1]

plt.scatter(fpr, tpr, c="red")
plt.plot(fpr, tpr, c="blue")
plt.xlabel("1 - Spec")
plt.ylabel("Sens")
plt.show()

print(roc_auc_score(y_test, y_pred_prob[:,1]))
print(log_loss(y_test, y_pred_prob[:,1]))



################## Glass ###############################3
from sklearn.preprocessing import LabelEncoder

glass = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Glass Identification\Glass.csv")
X = glass.drop('Type', axis=1)
y = glass['Type']
X = glass.drop('Type', axis=1)
y = glass['Type']
lbl = LabelEncoder()
y = lbl.fit_transform(y)
print(lbl.classes_)

X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,
                                                    test_size=0.3,
                                                    random_state=23)
nb = GaussianNB()
# Apriori Probabilities calculation
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print(classification_report(y_test, y_pred))
y_pred_prob = nb.predict_proba(X_test)

print(log_loss(y_test, y_pred_prob))

dtc = DecisionTreeClassifier(random_state=23,
                             max_depth=3)
dtc.fit(X_train, y_train)
y_pred_prob = dtc.predict_proba(X_test)

print(log_loss(y_test, y_pred_prob))


##########################################
comp_prob = pd.read_csv("comp_prob.csv")

print(roc_auc_score(comp_prob['y_test'],
                    comp_prob['yprob_1']))
print(roc_auc_score(comp_prob['y_test'],
                    comp_prob['yprob_2']))

print(log_loss(comp_prob['y_test'],
                    comp_prob['yprob_1']))
print(log_loss(comp_prob['y_test'],
                    comp_prob['yprob_2']))
