import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
import seaborn as sns

train = pd.read_csv("train.csv")

sch_hol = train[train['SchoolHoliday']==1]
no_sch_hol = train[train['SchoolHoliday']==0]

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
    
box_outliers(column='Sales', dataframe=sch_hol)
box_outliers(column='Sales', dataframe=no_sch_hol)

########################################
wine = pd.read_csv("wine.csv")

plt.figure(figsize=(15,10))
sns.heatmap(wine.drop('Class',axis=1).corr(),
            annot=True)
plt.show()

sns.scatterplot(data=wine, x='Phenols', y='Flavanoids',
                hue='Class')
plt.show()

###############################
from scipy.stats import norm

norm.sf(2000, 1678, 500)

norm.ppf(0.9, 1678, 500)

norm.cdf(1900, 1678, 500) - norm.cdf(1000, 1678, 500)

#####################################
x = np.array([0,1,2,3,4,5])
p = np.array([0.3,0.2,0.15,0.1,0.13,0.12])


e_x = np.sum(x*p)
print("E(X) =",e_x)

def simulate(n, x, p):
    np.random.seed(87)
    samp = np.random.rand(n)
    cp = np.cumsum(p)
    sim_data = []
    for r in samp:
        ind_sim = np.sum(cp < r)
        sim_data.append(x[ind_sim])
    return sim_data

s_data = simulate(n=30, 
         x=np.array([0,1,2,3,4,5]), 
         p=np.array([0.3,0.2,0.15,0.1,0.13,0.12]))

reserved = 2
s_data = np.array(s_data)

days = np.sum(s_data > reserved)
print("Days buses did not run =", days)

buses = np.where(s_data - reserved<0, 0,
                 s_data - reserved)
drivers = pd.DataFrame({'Simulated':s_data,
                        'Buses_Cancelled':buses})
print(drivers)
print("Buses did not run =", np.sum(buses))


################ Normality ######################
from scipy import stats
wine = pd.read_csv("wine.csv")

cols = wine.columns[1:]
print(stats.shapiro(wine['Alcohol'])[1])

p_vals = [stats.shapiro(wine[column])[1] for column in cols]

print(dict(zip(cols,p_vals)))

print(stats.shapiro(train['Sales'])[1])

################ Mann Whitney #################

# H0: distribution_0 = distrbution_1
# H1: distribution_0 ne distrbution_1
sch_0 = train[train['SchoolHoliday']==0]
sch_1 = train[train['SchoolHoliday']==1]
stats.mannwhitneyu(sch_0['Sales'], sch_1['Sales'])
## Conclusion: Distributions may not be equal

g = sns.FacetGrid(train, col="SchoolHoliday")
g.map(plt.hist, "Sales")
plt.show()

g = sns.FacetGrid(train, col="SchoolHoliday")
g.map(sns.kdeplot, "Sales")
plt.show()

############ ANOVA ################
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from scipy import stats 

wine['Class'] = wine['Class'].astype(str)
wine_ols = ols('Alcalinity ~ Class', data=wine).fit()
table = anova_lm(wine_ols, typ=2)
print(table)

sns.boxplot(data=wine, x='Class', y='Alcalinity')
plt.show()

g = sns.FacetGrid(wine, col="Class")
g.map(plt.hist, "Alcalinity")
plt.show()

cls1 = wine[wine['Class']==1]
cls2 = wine[wine['Class']==2]
cls3 = wine[wine['Class']==3]

stats.f_oneway(cls1['Alcalinity'],
               cls2['Alcalinity'],
               cls3['Alcalinity'])

############### Chi-square ######################
cars2018 = pd.read_csv("cars2018.csv")
ctab = pd.crosstab(cars2018['Transmission'], 
            cars2018['Aspiration'])
stats.chi2_contingency(ctab, correction=False)

molten = pd.melt(ctab.reset_index(), 
                 id_vars='Transmission')
sns.barplot(data=molten, x='Transmission',
            y='value',
            hue='Aspiration')
plt.show()


ctab = pd.crosstab(cars2018['Aspiration'], 
            cars2018['Drive'])
stats.chi2_contingency(ctab, correction=False)

ctab = pd.crosstab(cars2018['Drive'], 
            cars2018['Recommended Fuel'])
stats.chi2_contingency(ctab, correction=False)

############### Independent Test#################
houses = pd.read_csv("Housing.csv")
ac = houses[houses['airco']=='yes']
no_ac = houses[houses['airco']=='no']

# H0: var_ac = var_no_ac
# H1: var_ac ne var_no_ac
stats.bartlett(ac['price'], no_ac['price'])
## Conclusion: Variances may not be equal


# H0: mean_ac = mean_no_ac
# H0: mean_ac ne mean_no_ac
stats.ttest_ind(ac['price'], no_ac['price'],
                equal_var=False)
## Conclusion: Means may not be equal.


# H0: mean_ac <= mean_no_ac
# H0: mean_ac > mean_no_ac
stats.ttest_ind(ac['price'], no_ac['price'],
                equal_var=False, alternative="greater")
## Conclusion: Mean price of AC may be greater than mean price of non AC

#################### Regr Models ##################
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor 

X = cars2018.drop(['Model','Model Index','MPG'], axis=1)
y = cars2018['MPG']

X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3,
                                                    random_state=23)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(mean_squared_error(y_test, y_pred))

depths = np.arange(2,20)
scores = []
for d in depths:
    dtr = DecisionTreeRegressor(random_state=23, max_depth=d)
    dtr.fit(X_train, y_train)
    y_pred = dtr.predict(X_test)
    scores.append(mean_squared_error(y_test, y_pred))
    
i_min = np.argmin(scores)
print("Best Depth =", depths[i_min])
print("Best Score =", scores[i_min])

dtr = DecisionTreeRegressor(random_state=23, 
                            max_depth=depths[i_min])
dtr.fit(X_train, y_train)
importances = dtr.feature_importances_
features = dtr.feature_names_in_

plt.barh(features, importances)
plt.title("Feature Importances Plot")
plt.show()

########## Class Models ####################
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

brupt = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Company Bankruptcy\data.csv")
print(brupt['Bankrupt?'].value_counts(normalize=True)*100)

X = brupt.drop('Bankrupt?', axis=1)
y = brupt['Bankrupt?']
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    stratify=y,
                                                    test_size=0.3,
                                                    random_state=23)
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print(accuracy_score(y_test, y_pred))


depths = np.arange(2,20)
scores = []
for d in depths:
    dtr = DecisionTreeClassifier(random_state=23, max_depth=d)
    dtr.fit(X_train, y_train)
    y_pred = dtr.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
    
i_max = np.argmax(scores)
print("Best Depth =", depths[i_max])
print("Best Score =", scores[i_max])

dtr = DecisionTreeClassifier(random_state=23, 
                            max_depth=depths[i_max])
dtr.fit(X_train, y_train)
importances = dtr.feature_importances_
features = dtr.feature_names_in_

plt.figure(figsize=(40,35))
plt.barh(features, importances)
plt.title("Feature Importances Plot")
plt.show()