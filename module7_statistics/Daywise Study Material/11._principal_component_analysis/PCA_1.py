import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns 
import matplotlib.pyplot as plt

iris = pd.read_csv("iris.csv")
prcomp = PCA()
X = iris.drop('Species', axis=1)
components = prcomp.fit_transform(X)
components = pd.DataFrame(components,
                          columns=['PC1','PC2',
                                   'PC3','PC4'])
tot_var = components.var().sum()
print((components.var()/tot_var)*100)
components['Species'] = iris['Species']
sns.scatterplot(data=components,x='PC1',y='PC2',
                hue='Species')
plt.show()

#######################Glass ######################

glass = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Glass Identification\Glass.csv")

prcomp = PCA()
X = glass.drop('Type', axis=1)
components = prcomp.fit_transform(X)

pc_cols = ['PC'+str(i) for i in range(1, X.shape[1]+1)]
components = pd.DataFrame(components,
                          columns=pc_cols)
tot_var = components.var().sum()
print((components.var()/tot_var)*100)
components['Type'] = glass['Type']

plt.figure(figsize=(10,8))
sns.scatterplot(data=components,x='PC1',y='PC2',
                hue='Type')
plt.show()


