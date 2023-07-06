import pandas as pd 
from scipy import stats 

#### Puromycin
puro = pd.read_csv("Puromycin.csv")
treated = puro[puro['state']=='treated']
untreated = puro[puro['state']=='untreated']

# Population is or not Normal
print(stats.shapiro(treated['rate']))
print(stats.shapiro(untreated['rate']))

# H0: distribution_trt = distrbution_untrt
# H1: distribution_trt ne distrbution_untrt
stats.mannwhitneyu(treated['rate'], 
               untreated['rate'])
## Conclusion: Distributions may be equal

############## Cell Phone Survey
cell = pd.read_csv("Cell Phone Survey.csv")
males = cell[cell['Gender']=='M']
females = cell[cell['Gender']=='F']

# Population is or not Normal
print(stats.shapiro(males['Value for the Dollar']))
print(stats.shapiro(females['Value for the Dollar']))


# H0: distribution_male = distrbution_female
# H1: distribution_male ne distrbution_female
stats.mannwhitneyu(males['Value for the Dollar'], 
               females['Value for the Dollar'])
## Conclusion: Distributions may be equal


######### Soporific
sop = pd.read_csv("Soporific.csv")

# Population is or not Normal
print(stats.shapiro(sop['Drug A']))
print(stats.shapiro(sop['Drug B']))

# H0: distribution_A = distribution_B
# H1: distribution_A ne distribution_B
stats.mannwhitneyu(sop['Drug A'], 
               sop['Drug B'])
## Conclusion:  Distributions may be equal