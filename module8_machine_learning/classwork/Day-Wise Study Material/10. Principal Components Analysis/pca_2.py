import numpy as np 
import pandas as pd 
from sklearn.decomposition import PCA 
import  matplotlib.pyplot as plt 
import seaborn as sns 

a = np.array([4, 11, 8, 10])

np.linalg.norm(a)

iris = pd.read_csv("iris.csv")
X = iris.drop('Species', axis=1)
prcomp = PCA()

comps = prcomp.fit_transform(X)
print(comps.shape)

comps = pd.DataFrame(comps,
                     columns=['PC1','PC2','PC3','PC4'])
print(comps.var())
cov_mat = np.cov(X, rowvar=False)
values, vectors =  np.linalg.eig(cov_mat)
print("Eigen Values:\n",values)

print(prcomp.explained_variance_)
total_var = np.sum(prcomp.explained_variance_)
print("%age variations explained:")
print(prcomp.explained_variance_ratio_*100)

X_Red_PC = comps[['PC1','PC2']]
y = iris['Species']

X_Red_PC.loc[:,'Species'] = iris['Species'].values

sns.scatterplot(data=X_Red_PC, x='PC1',y='PC2',
                hue='Species')
plt.show()

############## Applying Supervised ##################
############### without PCs #########################
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.svm import SVC 
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    stratify=y,
                                                    random_state=23,
                                                    test_size=0.3)
svm = SVC(probability=True, kernel='linear', random_state=23)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_prob = svm.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

######################## With PCs ######################
X_Red_PC = comps[['PC1','PC2']]
y = iris['Species']
X_train, X_test, y_train, y_test = train_test_split(X_Red_PC,y,
                                                    stratify=y,
                                                    random_state=23,
                                                    test_size=0.3)
svm = SVC(probability=True, kernel='linear', random_state=23)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_prob = svm.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

# ##############################################################
# from sklearn.preprocessing import StandardScaler
# from pca import pca
# milk = pd.read_csv("milk.csv",index_col=0)
# scaler = StandardScaler()
# milk_scaled = scaler.fit_transform(milk)
# milk_scaled = pd.DataFrame(milk_scaled,
#                       columns=milk.columns,
#                       index=milk.index)
# model = pca()
# results = model.fit_transform(milk_scaled)


# model.biplot(label=True,legend=False)

# plt.show()
