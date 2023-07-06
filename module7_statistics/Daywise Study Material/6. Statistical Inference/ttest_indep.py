import pandas as pd 
from scipy import stats 

co2 = pd.read_csv("CO2.csv")

nonchilled = co2[co2['Treatment']=='nonchilled']
chilled = co2[co2['Treatment']=='chilled']

# H0: var_chilled = var_nonchilled
# H1: var_chilled ne var_nonchilled
stats.bartlett(chilled['uptake'], nonchilled['uptake'])
## Conclusion: Variances may be equal

# H0: mean_chilled = mean_nonchilled
# H0: mean_chilled ne mean_nonchilled
stats.ttest_ind(chilled['uptake'], nonchilled['uptake'],
                equal_var=True)
## Conclusion: Means may not be equal.

############## Cell Phone Survey
cell = pd.read_csv("Cell Phone Survey.csv")
males = cell[cell['Gender']=='M']
females = cell[cell['Gender']=='F']

# H0: var_males = var_fem
# H1: var_males ne var_fem
stats.bartlett(males['Value for the Dollar'], 
               females['Value for the Dollar'])
## Conclusion: Variances may be equal

# H0: mean_males = mean_fem
# H1: mean_males ne mean_fem
stats.ttest_ind(males['Value for the Dollar'], 
               females['Value for the Dollar'],
                equal_var=True)
## Conclusion: Means may be equal

######## Puromycin
puro = pd.read_csv("Puromycin.csv")
treated = puro[puro['state']=='treated']
untreated = puro[puro['state']=='untreated']

# H0: var_trt = var_untrt
# H1: var_trt ne var_untrt
stats.bartlett(treated['rate'], 
               untreated['rate'])
## Conclusion: Variances may be equal

# H0: mean_trt = mean_untrt
# H1: mean_trt ne mean_untrt
stats.ttest_ind(treated['rate'], 
               untreated['rate'],
                equal_var=True)
## Conclusion: Means may be equal

######### Soporific
sop = pd.read_csv("Soporific.csv")

# H0: var_A = var_B
# H1: var_A ne var_B
stats.bartlett(sop['Drug A'], 
               sop['Drug B'])
## Conclusion: Variances may be equal

# H0: mean_A = mean_B
# H1: mean_A ne mean_B
stats.ttest_ind(sop['Drug A'], 
               sop['Drug B'],
                equal_var=True)
## Conclusion: Means may be equal