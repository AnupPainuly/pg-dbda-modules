import pandas as pd
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

######################Example 1#################################
agr = pd.read_csv("Yield.csv")
agrYield = ols('Yield ~ Treatments', data=agr).fit()
table = anova_lm(agrYield, typ=2)
print(table)

######## Post Hoc Tukey HSD ################
compare = pairwise_tukeyhsd(agr['Yield'], 
                            agr['Treatments'], alpha=0.05)

dd = pd.DataFrame(compare._results_table.data[1:],
                  columns=compare._results_table.data[0])
dd



#agr.groupby('Treatments')['Yield'].mean()

##### GMAT
gmat = pd.read_csv("GMAT Scores.csv")
gmat_ols = ols('Score ~ Major', data=gmat).fit()
table = anova_lm(gmat_ols, typ=2)
print(table)

compare = pairwise_tukeyhsd(gmat['Yield'], 
                            agr['Treatments'], alpha=0.05)

dd = pd.DataFrame(compare._results_table.data[1:],
                  columns=compare._results_table.data[0])
dd

####### Funds
funds = pd.read_csv("funds.csv")
funds_ols = ols('Expense_Ratio ~ Fund', data=funds).fit()
table = anova_lm(funds_ols, typ=2)
print(table)

compare = pairwise_tukeyhsd(funds['Expense_Ratio'], 
                            funds['Fund'], alpha=0.05)

dd = pd.DataFrame(compare._results_table.data[1:],
                  columns=compare._results_table.data[0])
dd

import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(data=funds, x='Fund', y='Expense_Ratio')
plt.show()



