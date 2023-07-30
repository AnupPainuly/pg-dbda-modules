import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

milk = pd.read_csv("milk.csv",index_col=0)

scaler = StandardScaler()
milkscaled=scaler.fit_transform(milk)

km = KMeans(n_clusters=2, random_state=23)
km.fit(milkscaled)
print(km.labels_)

sil = []
clust = np.arange(2,9)
for i in clust:
    clustering = KMeans(n_clusters=i, random_state=23)
    clustering.fit(milkscaled)
    sil.append(silhouette_score(milkscaled,
                                clustering.labels_))

i_max = np.argmax(sil)
print("Best no. of clusters =",clust[i_max])

#### clustering with best k

km = KMeans(n_clusters=clust[i_max], random_state=23)
km.fit(milkscaled)
print(km.labels_)

milk['Clust'] = km.labels_

milk.sort_values(by='Clust')

milk.groupby('Clust').mean()

############### Nutrient #####################

nut = pd.read_csv("nutrient.csv",index_col=0)

scaler = StandardScaler()
nutscaled=scaler.fit_transform(nut)


sil = []
clust = np.arange(2,9)
for i in clust:
    clustering = KMeans(n_clusters=i, random_state=23)
    clustering.fit(nutscaled)
    sil.append(silhouette_score(nutscaled,
                                clustering.labels_))

i_max = np.argmax(sil)
print("Best no. of clusters =",clust[i_max])

#### clustering with best k

km = KMeans(n_clusters=clust[i_max], random_state=23)
km.fit(nutscaled)
print(km.labels_)

nut['Clust'] = km.labels_

nut.sort_values(by='Clust')

nut.groupby('Clust').mean()

###############  RFM ###########################
rfm = pd.read_csv("rfm_data_customer.csv", index_col=0)
rfm.drop('most_recent_visit', axis=1, inplace=True)

scaler = StandardScaler()
rfmscaled=scaler.fit_transform(rfm)

sil = []
clust = np.arange(2,9)
for i in clust:
    clustering = KMeans(n_clusters=i, random_state=23)
    clustering.fit(rfmscaled)
    sil.append(silhouette_score(rfmscaled,
                                clustering.labels_))

i_max = np.argmax(sil)
print("Best no. of clusters =",clust[i_max])

#### clustering with best k

km = KMeans(n_clusters=3, random_state=23)
km.fit(rfmscaled)
print(km.labels_)

rfm['Clust'] = km.labels_

rfm.sort_values(by='Clust')

rfm.groupby('Clust').mean()

#################################################
from statsmodels.datasets import get_rdataset
USArrests = get_rdataset('USArrests').data

scaler = StandardScaler()
rfmscaled=scaler.fit_transform(USArrests)

sil = []
clust = np.arange(2,9)
for i in clust:
    clustering = KMeans(n_clusters=i, random_state=23)
    clustering.fit(rfmscaled)
    sil.append(silhouette_score(rfmscaled,
                                clustering.labels_))

i_max = np.argmax(sil)
print("Best no. of clusters =",clust[i_max])

#### clustering with best k

km = KMeans(n_clusters=2, random_state=23)
km.fit(rfmscaled)
print(km.labels_)

USArrests['Clust'] = km.labels_

USArrests.sort_values(by='Clust')

USArrests.groupby('Clust').mean()

########### Elbow Method ###################
clustering = KMeans(n_clusters=24, random_state=23)
clustering.fit(milkscaled)
print(clustering.inertia_)


wss = []
clust = np.arange(2,11)
for i in clust:
    clustering = KMeans(n_clusters=i, random_state=23)
    clustering.fit(milkscaled)
    wss.append(clustering.inertia_)

plt.scatter(clust, wss, color='red')
plt.plot(clust, wss, color='blue')
plt.xlabel("No. of Clusters")
plt.ylabel("WSS")
plt.title("Scree Plot")
plt.show()





wss = []
clust = np.arange(2,11)
for i in clust:
    clustering = KMeans(n_clusters=i, random_state=23)
    clustering.fit(nutscaled)
    wss.append(clustering.inertia_)

plt.scatter(clust, wss, color='red')
plt.plot(clust, wss, color='blue')
plt.xlabel("No. of Clusters")
plt.ylabel("WSS")
plt.title("Scree Plot")
plt.show()

