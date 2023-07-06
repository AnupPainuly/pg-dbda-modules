import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 

cars93 = pd.read_csv("Cars93.csv")

cars93['Price'].plot(kind='box')
plt.show()

q1 = np.quantile(cars93['Price'], 0.25)
q3 = np.quantile(cars93['Price'], 0.75)
iqr = q3 - q1 
print("Inter-Quartile Range =", iqr)
print("1.5 IQR =", 1.5*iqr)

upper = cars93[cars93['Price'] > 1.5*iqr + q3 ]
upper['Price']

lower = cars93[cars93['Price'] < q1 - 1.5*iqr ]
lower['Price']

def box_outliers(column, dataframe):  
    q1 = np.quantile(dataframe[column].dropna(), 0.25)
    q3 = np.quantile(dataframe[column].dropna(), 0.75)
    iqr = q3 - q1 
    upper = dataframe[dataframe[column] > 1.5*iqr + q3 ]
    print("Upper Outliers :")
    print(list(upper[column]))
    
    lower = dataframe[dataframe[column] < q1 - 1.5*iqr ]
    print("Lower Outliers =") 
    print(list(lower[column]))
    
    plt.boxplot(dataframe[column].dropna())
    plt.title("Boxplot of " + column)
    plt.show()

    
box_outliers('Price', cars93)    
box_outliers('Weight', cars93)

cars93['MPG.highway'].plot(kind='box')
plt.show()

box_outliers('MPG.highway', cars93)

## Air Quality
air = pd.read_csv("Airquality.csv")

box_outliers('Ozone', air)  
plt.boxplot(air['Ozone'].dropna())
plt.show()

box_outliers('Solar.R', air)  
plt.boxplot(air['Solar.R'].dropna())
plt.show()

box_outliers('Wind', air)  
plt.boxplot(air['Wind'].dropna())
plt.show()

box_outliers('Temp', air)  
plt.boxplot(air['Temp'].dropna())
plt.show()

################## iris
iris = pd.read_csv("iris.csv")

for col in ['Sepal.Length', 'Sepal.Width', 
            'Petal.Length', 'Petal.Width'] : 
    print("Column =", col)
    box_outliers(col, iris)

####### Salaries
salaries = pd.read_csv("Salaries.csv")
sals_num = salaries.select_dtypes(exclude='object')

for col in sals_num.columns : 
    print("Column =", col)
    box_outliers(col, sals_num)
    

####### Nutrient
nutrient = pd.read_csv("nutrient.csv")
nutrient_num = nutrient.select_dtypes(exclude='object')

for col in nutrient_num.columns : 
    print("Column =", col)
    box_outliers(col, nutrient_num)
    