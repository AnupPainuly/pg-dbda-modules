import pandas as pd 
import numpy as np 
from scipy import stats

plant = pd.read_csv("PlantGrowth.csv")
s = np.std(plant['weight'], ddof = 1)
xbar = np.mean(plant['weight'])
t = (xbar - 6)/(s/np.sqrt(30))
print("Test Statistic =", t)

# H0: mu = 6
# H1: mu ne 6
stats.ttest_1samp(plant['weight'],popmean=6.0,
                  alternative='two-sided')

# H0: mu <= 6
# H1: mu >  6
stats.ttest_1samp(plant['weight'],popmean=6.0,
                  alternative='greater')

# H0: mu >= 6
# H1: mu  < 6
stats.ttest_1samp(plant['weight'],popmean=6.0,
                  alternative='less')

## Conclusion: Population Mean may be less than 6

################ Consumer Transporation
import os 
os.chdir(r"C:\Training\Reference Books\Data_Files")
consumer = pd.read_excel("Consumer Transportation Survey.xlsx",
                         skiprows=2)
# a. H0: Hours per week >= 8    Vs   H1: Hours per week < 8
stats.ttest_1samp(consumer['# of hours per week in vehicle'],
                  popmean=8.0,
                  alternative='less')
## Conclusion: Hours per week may be at least 8.

# b. H0: Miles per week = 600  Vs  H1: Miles per week
stats.ttest_1samp(consumer['Miles driven per week'],
                  popmean=600,
                  alternative='two-sided')
## Conclusion: Miles per week may not be 600

# c. H0: Age of SUV <= 35  Vs H1: Age of SUV > 35
suv = consumer[consumer['Vehicle Driven']=="SUV"]
stats.ttest_1samp(suv['Age'],
                  popmean=35,
                  alternative='greater')
## Conclusion: Age of SUV driver may be greater than 35

######## Sales Data
sales = pd.read_excel("Sales Data.xlsx", skiprows=2)
# H0: avg profit per customer <= 4500
# H1: avg profit per customer  > 4500
stats.ttest_1samp(sales['Gross Profit'].dropna(),
                  popmean=4500, alternative='greater')
## Conclusion: avg profit per customer may not be greater than 4500

########## Airport Service times
airport = pd.read_excel("Airport Service Times.xlsx",
                        usecols="A", skiprows=2)
# H0: Times >= 150  Vs  H1: Times < 150
stats.ttest_1samp(airport['Times (sec.)'].dropna(),
                  popmean=150, alternative='less')
## Conclusion: Service time may be less than 2.5 minutes