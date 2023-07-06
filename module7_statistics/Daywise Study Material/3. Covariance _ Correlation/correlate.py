import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 

x = np.array([23, 56, 78, 90, 109, 123])
y = np.array([789, 896, 908, 1023, 1348, 1789])

covariance = np.mean((x-np.mean(x))*(y-y.mean()))
print("Covariance =", covariance)
std_x = np.std(x)
std_y = np.std(y)
correlation = covariance/(std_x*std_y)
print("Correlation =", correlation)

np.corrcoef(x, y)

print(np.cov(x,y,ddof=0))

y = np.array([567, 345,290,134, 109, 56])
covariance = np.mean((x-np.mean(x))*(y-y.mean()))
print("Covariance =", covariance)
std_x = np.std(x)
std_y = np.std(y)
correlation = covariance/(std_x*std_y)
print("Correlation =", correlation)

np.corrcoef(x, y)

print(np.cov(x,y,ddof=0))


################## iris
iris = pd.read_csv("iris.csv")

iris.cov(ddof=0)
iris.corr()

sns.pairplot(data=iris)
plt.show()

sns.heatmap(iris.corr(),
            annot=True)
plt.show()


####### Nutrient
nutrient = pd.read_csv("nutrient.csv")


sns.pairplot(data=nutrient)
plt.show()

sns.heatmap(nutrient.corr(),
            annot=True)
plt.show()

########### bank
bank = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\bank\bank.csv",
                   sep=';')
bank.info()

sns.pairplot(data=bank)
plt.show()

sns.heatmap(bank.corr(),
            annot=True)
plt.show()

sns.scatterplot(data=bank, x='pdays', y='previous',
                hue='y')
plt.show()

###### Concrete
concrete = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Concrete Strength\Concrete_Data.csv")

sns.pairplot(data=concrete)
plt.show()

sns.heatmap(concrete.corr(),
            annot=True)
plt.show()
