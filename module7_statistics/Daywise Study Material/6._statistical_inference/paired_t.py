import pandas as pd 
import numpy as np 
from scipy import stats

anorexia = pd.read_csv("anorexia.csv")
cont = anorexia[anorexia['Treat']=='Cont']
cbt = anorexia[anorexia['Treat']=='CBT']
ft = anorexia[anorexia['Treat']=='FT']
# H0: D >= 0  Vs  H1: D < 0
stats.ttest_rel(cont['Prewt'], cont['Postwt'],
                alternative="less")
## Conclusion: Treatment may not be effective.

# H0: D >= 0  Vs  H1: D < 0
stats.ttest_rel(cbt['Prewt'], cbt['Postwt'],
                alternative="less")
## Conclusion: Treatment may be effective.

# H0: D >= 0  Vs  H1: D < 0
stats.ttest_rel(ft['Prewt'], ft['Postwt'],
                alternative="less")
## Conclusion: Treatment may be effective.

#######################################
import os 
os.chdir(r"C:\Training\Reference Books\Data_Files")
ohio = pd.read_csv("Ohio Education Performance.csv")

# H0: D = 0  Vs  H1: D ne 0
stats.ttest_rel(ohio['Writing'], ohio['Reading'])
## Conclusion: Writing and reading may not be equal

# H0: D = 0  Vs  H1: D ne 0
stats.ttest_rel(ohio['Math'], ohio['Science'])
## Conclusion: Math and Science may not be equal

rhum = pd.read_csv("Rheumatic.csv")
# H0: D >= 0  Vs  H1: D < 0
stats.ttest_rel(rhum['Before'], rhum['After'],
                alternative="less")
## Conclusion: Treatment may be effective

# H0: D <= 0  Vs  H1: D > 0
stats.ttest_rel(rhum['After'], rhum['Before'],
                alternative="greater")

