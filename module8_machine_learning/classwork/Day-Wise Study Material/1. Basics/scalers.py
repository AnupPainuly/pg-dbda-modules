import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
a = np.array([[100,0.1],
            [89,0.67],
            [56,0.4],
            [49,0.9]])

pa = pd.DataFrame(a, 
                  columns=['x1','x2'])
scaler = StandardScaler()
### Calculates mean and std deviation of each column
scaler.fit(pa)

print(pa.mean())
print(pa.std(ddof=0))

print(scaler.mean_)
print(scaler.scale_)
### It actually transforms the data 
scaler.transform(pa)

#### Min Max
a = np.array([[100,0.1],
            [89,0.67],
            [56,0.4],
            [49,0.9]])

pa = pd.DataFrame(a, 
                  columns=['x1','x2'])
scaler = MinMaxScaler()
### Calculates Min and Max of each column
scaler.fit(pa)

print(pa.min())
print(pa.max())

print(scaler.data_min_)
print(scaler.data_max_)
### It actually transforms the data 
scaler.transform(pa)
