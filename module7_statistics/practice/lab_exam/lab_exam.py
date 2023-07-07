#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os

os.chdir(r"/home/darkstar/Documents/pg-dbda/module7_statistics/Daywise Study Material/datasets")


# In[71]:


df = pd.read_csv("austin_waste_and_diversion.csv",sep="\t")
df


# In[26]:


filtered_df = df[df['route_number'] == "BR01"]
filtered_df


# In[ ]:


## A. Mean, median, mode for load_weight column


# In[28]:


np.mean(filtered_df['load_weight'])


# In[29]:


np.median(filtered_df['load_weight'])


# In[33]:


stats.mode(filtered_df['load_weight'])


# In[ ]:


## #B. Skewness, kurtosis for load_weight column


# In[34]:


skewness = df['load_weight'].skew()
kurtosis = df['load_weight'].kurtosis()
print(skewness)
print(kurtosis)


# In[ ]:


#C Plot Histogram for load_weight column


# In[49]:


# Plot histogram
plt.hist(filtered_df['load_weight'], bins=10, edgecolor='black')

# Set labels and title
plt.xlabel('Load Weight')
plt.ylabel('Frequency')
plt.title('Histogram of Load Weight')


# In[ ]:





# In[ ]:


#D Plot boxplot for load_weight column


# In[48]:


sns.boxplot(x=df['load_weight'])
plt.title('Boxplot of a load weight')

# Show the plot
plt.show()


# In[57]:


value_counts = df['dropoff_site'].value_counts()
value_counts.plot.pie(autopct='%1.1f%%')
# Set title
plt.title('Drop Off Sites Distribution')

# Show the plot
plt.show()


# #Q2
# #A. Apply approprirate hypothesis to compare mean of load weight for following two grups first group:: 
# #when load_type = BRUSH, second group:: when load_type  = BULK
# #Selecting rieght hypothesis test
# #syntax and finding  result of hypothesis
# #Conclusion

# In[60]:


# Split the data into two groups based on load_type
group1 = df[df['load_type'] == 'BRUSH']['load_weight'].dropna()
group2 = df[df['load_type'] == 'BULK']['load_weight'].dropna()

# Perform two-sample t-test
t_stat, p_value = stats.ttest_ind(group1, group2)

print("t-statistic:", t_stat)
print("p-value:", p_value)


# Null hypothesis (H0): There is no significant difference in the mean load weight between the "BRUSH" and "BULK" load types.
# Alternative hypothesis (H1): There is a significant difference in the mean load weight between the "BRUSH" and "BULK" load types.
# 
# With a p-value close to zero, you would reject the null hypothesis (H0) and accept the alternative hypothesis (H1). This implies that the "BRUSH" and "BULK" load types have statistically different mean load weights.
# 

# #B. Select any 3 different route numners. Explain the impact of route number on load weight using
# #appropriate hypothesis test. (ex. BR01, BR02, BR03 OR any three route numbers of your choice)
# #Selecting rieght hypothesis test
# #syntax and finding  result of hypothesis
# #Conclusion

# In[61]:


#ANOVA
result = stats.f_oneway(df[df['route_number'] == 'BR01']['load_weight'],
                        df[df['route_number'] == 'BR02']['load_weight'],
                        df[df['route_number'] == 'BR03']['load_weight'])

# Extract test statistic and p-value
f_stat, p_value = result.statistic, result.pvalue

print("F-statistic:", f_stat)
print("p-value:", p_value)


# Null hypothesis (H0): There is no significant impact of the route number on the load weight.
# Alternative hypothesis (H1): There is a significant impact of the route number on the load weight.
#     
# p-value is 0.006, which is below the typical significance level of 0.05, we reject the null hypothesis (H0). The null hypothesis assumes that there is no significant impact of the route number on the load weight.
# 
# Therefore, based on the observed data, we accept the alternative hypothesis (H1). The alternative hypothesis suggests that there is a statistically significant impact of the route number on the load weight.

# In[68]:


df = pd.read_excel("Closing Stock Prices.xlsx")
df

#Extract the IBM, INTC, and GE column data
ibm_data = df['IBM']
intel_data = df['INTC']
ge_data = df['GE']

# Perform Shapiro-Wilk test for normality
_, ibm_pvalue = stats.shapiro(ibm_data)
_, intel_pvalue = stats.shapiro(intel_data)
_, ge_pvalue = stats.shapiro(ge_data)

# Print p-values
print("IBM p-value:", ibm_pvalue)
print("INTC p-value:", intel_pvalue)
print("GE p-value:", ge_pvalue)

# Plot histograms for visual inspection
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.hist(ibm_data, bins='auto')
plt.xlabel('IBM Closing Stock Prices')
plt.subplot(1, 3, 2)
plt.hist(intel_data, bins='auto')
plt.xlabel('INTC Closing Stock Prices')
plt.subplot(1, 3, 3)
plt.hist(ge_data, bins='auto')
plt.xlabel('GE Closing Stock Prices')
plt.tight_layout()
plt.show()


# #q3 given data:: Closing Stock Prices.xlsx
# #A. Check normality of IBM. INTC and GE column data

# In[65]:


import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

# Read the Excel file
df = pd.read_excel('Closing Stock Prices.xlsx')

# Extract the IBM, INTC, and GE column data
ibm_data = df['IBM']
intel_data = df['INTC']
ge_data = df['GE']

# Perform Shapiro-Wilk test for normality
_, ibm_pvalue = stats.shapiro(ibm_data)
_, intel_pvalue = stats.shapiro(intel_data)
_, ge_pvalue = stats.shapiro(ge_data)

# Print p-values
print("IBM p-value:", ibm_pvalue)
print("INTC p-value:", intel_pvalue)
print("GE p-value:", ge_pvalue)

# Create Q-Q plots for visual inspection
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
stats.probplot(ibm_data, dist="norm", plot=plt)
plt.title("IBM Q-Q Plot")
plt.subplot(1, 3, 2)
stats.probplot(intel_data, dist="norm", plot=plt)
plt.title("INTC Q-Q Plot")
plt.subplot(1, 3, 3)
stats.probplot(ge_data, dist="norm", plot=plt)
plt.title("GE Q-Q Plot")
plt.tight_layout()
plt.show()


# In[ ]:


#example of normally distributed


# In[66]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Generate normally distributed data
np.random.seed(0)
data = np.random.normal(loc=0, scale=1, size=1000)

# Create Q-Q plot
stats.probplot(data, dist="norm", plot=plt)

# Set plot title
plt.title("Q-Q Plot - Normally Distributed Data")

# Show the plot
plt.show()


# In[75]:


df = pd.read_excel('Closing Stock Prices.xlsx')
df = df.rename(columns={'DJ Industrials \nIndex': 'DJ Industrial Index'})
df


# #B. Apply multiple regression and predict "DJ industrial index" using IBM, INTC, CSCO and GE columns
# #Create a heat map of correlation matrix
# #Write equation of the create regression line

# In[79]:


import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_excel('Closing Stock Prices.xlsx')
df = df.rename(columns={'DJ Industrials \nIndex': 'DJ Industrial Index'})

# Extract the independent variables (IBM, INTC, CSCO, GE) and dependent variable (DJ Industrial Index)
X = df[['IBM', 'INTC', 'CSCO', 'GE']]
y = df['DJ Industrial Index']

# Fit the multiple regression model
regression_model = LinearRegression()
regression_model.fit(X, y)

# Perform predictions
y_pred = regression_model.predict(X)

# Calculate the coefficients and intercept of the regression line
coefficients = regression_model.coef_
intercept = regression_model.intercept_

# Print the equation of the regression line
equation = "DJ Industrial Index = "
for i in range(len(coefficients)):
    equation += f"({coefficients[i]:.2f} * {X.columns[i]}) + "
equation += f"{intercept:.2f}"
print("Equation of the regression line:", equation)

# Create a correlation matrix heatmap
corr_matrix = df[['IBM', 'INTC', 'CSCO', 'GE', 'DJ Industrial Index']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix Heatmap")
plt.show()


# In[ ]:




