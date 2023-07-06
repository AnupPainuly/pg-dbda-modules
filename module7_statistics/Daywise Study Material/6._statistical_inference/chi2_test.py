import numpy as np
from scipy.stats import chi2_contingency
import pandas as pd 

obs = np.array([[9,4],
                [6,9]])
exp = np.array([[6.96,6.03],
               [8.03,6.96]])

chi_sq = np.sum(((obs.ravel()-exp.ravel())**2)/exp.ravel())
print("Statistic =", chi_sq)

print(chi2_contingency(obs,correction=False))

### 49

obs = np.array([[13,7], [12,22]])

print(chi2_contingency(obs,correction=False))


#### 55
acc = pd.read_csv("New Account Processing.csv")

a = pd.crosstab(index=acc['Gender'], 
                columns=acc['Certified'])
print(chi2_contingency(a,correction=False))

b = pd.crosstab(index=acc[' Prior Background'], 
                columns=acc['Certified'])
print(chi2_contingency(b, correction=False))

