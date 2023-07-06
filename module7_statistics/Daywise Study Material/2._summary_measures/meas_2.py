import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

air = pd.read_csv("Airquality.csv")

######### Ozone
plt.hist(air['Ozone'], bins=10)
plt.show()

plt.boxplot(air['Ozone'].dropna())
plt.show()

print(np.quantile(air['Ozone'].dropna(), 
                  [0.25,0.5,0.75]))
print(air['Ozone'].skew())
print(air['Ozone'].kurtosis())

#### Wind
plt.hist(air['Wind'], bins=10)
plt.show()
print(air['Wind'].skew())
print(air['Wind'].kurtosis())


plt.boxplot(air['Wind'].dropna())
plt.show()

print(np.quantile(air['Wind'].dropna(), 
                  [0.25,0.5,0.75]))

####### Temp
plt.hist(air['Temp'], bins=10)
plt.xlabel('Temperature')
plt.title('Histogram')
plt.show()
print(air['Temp'].skew())
print(air['Temp'].kurtosis())

plt.boxplot(air['Temp'].dropna())
plt.ylabel('Temperature')
plt.title('Boxplot')
plt.show()

print(np.quantile(air['Temp'].dropna(), 
                  [0.25,0.5,0.75]))


diamonds = pd.read_csv("diamonds.csv")
plt.hist(diamonds['price'], bins=10)
plt.xlabel('Price')
plt.title('Histogram')
plt.show()
print(diamonds['price'].skew())
print(diamonds['price'].kurtosis())

################## Iris ####################
iris = pd.read_csv("iris.csv")
print(iris['Sepal.Length'].skew())
print(iris['Sepal.Length'].kurtosis())

plt.boxplot(iris['Sepal.Length'].dropna())
plt.ylabel('Sepal Length')
plt.title('Boxplot')
plt.show()

plt.hist(iris['Sepal.Length'].dropna(), bins=10)
plt.xlabel('Sepal Length')
plt.title('Histogram')
plt.show()

## setosa
setosa = iris[iris['Species']=='setosa']
print(setosa['Sepal.Length'].skew())
print(setosa['Sepal.Length'].kurtosis())
print(np.quantile(setosa['Sepal.Length'],
                  [0.25, 0.5, 0.75]))
plt.boxplot(setosa['Sepal.Length'])
plt.title("Setosa")
plt.show()
## versicolor
versicolor = iris[iris['Species']=='versicolor']
print(versicolor['Sepal.Length'].skew())
print(versicolor['Sepal.Length'].kurtosis())
print(np.quantile(versicolor['Sepal.Length'],
                  [0.25, 0.5, 0.75]))
plt.boxplot(versicolor['Sepal.Length'])
plt.title("Versicolor")
plt.show()
## virginica
virginica = iris[iris['Species']=='virginica']
print(virginica['Sepal.Length'].skew())
print(virginica['Sepal.Length'].kurtosis())
print(np.quantile(virginica['Sepal.Length'],
                  [0.25, 0.5, 0.75]))
plt.boxplot(virginica['Sepal.Length'])
plt.title("Virginica")
plt.show()

import seaborn as sns
sns.boxplot(x='Species', y='Sepal.Length', data=iris)
plt.show()

sns.boxplot(x='Species', y='Sepal.Width', data=iris)
plt.show()

sns.boxplot(x='Species', y='Petal.Length', data=iris)
plt.show()

sns.boxplot(x='Species', y='Petal.Width', data=iris)
plt.show()



### Facet Grid
g = sns.FacetGrid(iris, col="Species")
g = g.map(plt.hist, "Sepal.Length")
plt.show()

g = sns.FacetGrid(iris, col="Species")
g = g.map(plt.hist, "Sepal.Width")
plt.show()










# ### Facet Grid
# g = sns.FacetGrid(iris, col="Species")
# g = g.map(plt.boxplot, "Sepal.Length")
# plt.show()



