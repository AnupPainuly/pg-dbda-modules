from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

milk = pd.read_csv("milk.csv",index_col=0)

scaler = StandardScaler()
milkscaled=scaler.fit_transform(milk)

mergings = linkage(milkscaled,method='average')
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=np.array(milk.index),
           leaf_rotation=45,
           leaf_font_size=10,
)
plt.show()


##################################################

clustering = AgglomerativeClustering(n_clusters=3,
                                     linkage='average') 
clustering.fit(milkscaled)

print(clustering.labels_)
print(silhouette_score(milkscaled,clustering.labels_))

sil = []
clust = np.arange(2,9)
for i in clust:
    clustering = AgglomerativeClustering(n_clusters=i,
                                         linkage='average') 
    clustering.fit(milkscaled)
    sil.append(silhouette_score(milkscaled,
                                clustering.labels_))

i_max = np.argmax(sil)
print("Best no. of clusters =",clust[i_max])

############ iris #######################
iris = pd.read_csv("iris.csv")
X = iris.drop('Species', axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


sil = []
clust = np.arange(2,9)
for i in clust:
    clustering = AgglomerativeClustering(n_clusters=i,
                                         linkage='complete') 
    clustering.fit(X_scaled)
    sil.append(silhouette_score(X_scaled,
                                clustering.labels_))

i_max = np.argmax(sil)
print("Best no. of clusters =",clust[i_max])
print("Best Score =", sil[i_max])
