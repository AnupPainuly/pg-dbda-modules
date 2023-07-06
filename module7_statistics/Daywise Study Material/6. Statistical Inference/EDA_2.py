import pandas as pd 
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

hr = pd.read_csv("HR_comma_sep.csv")

## Is Satisfaction related to Department?
hr_ols = ols('satisfaction_level ~ Department', 
             data=hr).fit()
table = anova_lm(hr_ols, typ=2)
print(table)
## Conclusion: Satisfaction may be related to Department
    
sns.boxplot(data=hr, x='Department',
            y='satisfaction_level')
plt.show()

cts = hr.groupby('Department')['satisfaction_level'].mean()
plt.barh(cts.index, cts)
plt.ylabel("Department")
plt.xlabel("satisfaction_level")
plt.show()


## Is last_evaluation related to Department?
hr_ols = ols('last_evaluation ~ Department', 
             data=hr).fit()
table = anova_lm(hr_ols, typ=2)
print(table)
## Conclusion: Satisfaction may not be related to Department
    
sns.boxplot(data=hr, x='Department',
            y='last_evaluation')
plt.show()

cts = hr.groupby('Department')['last_evaluation'].mean()
plt.barh(cts.index, cts)
plt.ylabel("Department")
plt.xlabel("last_evaluation")
plt.show()

# Is leaving and Department related?
ctab = pd.crosstab(index=hr['Department'],
                   columns=hr['left'])
print(chi2_contingency(ctab,correction=False))
# Conclusion: leaving and Department may be related


df_bar = pd.melt(ctab.reset_index(),id_vars="Department")
sns.barplot(y="Department",
           x="value",
           hue="left",
           data=df_bar)
plt.title("Grouped Bar Chart")
plt.show()

# Is leaving and promo in 5 yrs related?
ctab = pd.crosstab(index=hr['promotion_last_5years'],
                   columns=hr['left'])
print(chi2_contingency(ctab,correction=False))
# Conclusion: leaving and promotion_last_5years may be related


df_bar = pd.melt(ctab.reset_index(),id_vars="promotion_last_5years")
sns.barplot(x="promotion_last_5years",
           y="value",
           hue="left",
           data=df_bar)
plt.title("Grouped Bar Chart")
plt.show()

