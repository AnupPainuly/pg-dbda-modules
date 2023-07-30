import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN 
from sklearn.metrics import silhouette_score

milk = pd.read_csv("milk.csv",index_col=0)

scaler = StandardScaler()
milkscaled=scaler.fit_transform(milk)

clustering = DBSCAN(eps=0.7, min_samples=2)
clustering.fit(milkscaled)
milk_cluster = pd.DataFrame(milkscaled,
                          columns=milk.columns)
milk_cluster['Cluster'] = clustering.labels_
milk_cluster = milk_cluster[clustering.labels_ != -1]
silhouette_score(milk_cluster.drop('Cluster', axis=1),
                 milk_cluster['Cluster'])


epsilons = np.linspace(0.01, 2, 20)
sil_score = {}
for epsilon in epsilons:
    clustering = DBSCAN(eps=epsilon, min_samples=2)
    clustering.fit(milkscaled)
    if len(np.unique(clustering.labels_)) > 2:
        milk_cluster = pd.DataFrame(milkscaled,
                                  columns=milk.columns)
        milk_cluster['Cluster'] = clustering.labels_
        milk_cluster = milk_cluster[clustering.labels_ != -1]
    
        sil = silhouette_score(milk_cluster.drop('Cluster', axis=1),
                         milk_cluster['Cluster'])
        sil_score[str(epsilon)] = sil
        
sil_data = []
for item in sil_score.items():
    sil_data.append(item)
    
df_score = pd.DataFrame(sil_data, columns=['eps','score'])
print("Best Params and Score:")
df_score.sort_values(by='score', ascending=False).iloc[0,:]


#### Fitting the best
clustering = DBSCAN(eps=0.3242105263157895, min_samples=2)
clustering.fit(milkscaled)
print(clustering.labels_)
milk['Clust'] = clustering.labels_
