import numpy as np
import pandas as pd

x = np.array([23, 56, 78, 90, 109, 123])
y = np.array([789, 896, 908, 1023, 1348, 1789])

covariance = np.mean((x-np.mean(x))*(y-y.mean()))
print("Covariance =", covariance)

print(np.cov(x,y,ddof=0))

y = np.array([567, 345,290,134, 109, 56])
covariance = np.mean((x-np.mean(x))*(y-y.mean()))
print("Covariance =", covariance)

print(np.cov(x,y,ddof=0))

################## iris
iris = pd.read_csv("iris.csv")

iris.cov(ddof=0)
