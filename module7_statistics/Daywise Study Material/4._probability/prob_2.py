import pandas as pd
import numpy as np

import os
os.chdir(r"C:\Training\Reference Books\Data_Files")

consumer = pd.read_excel("Consumer Transportation Survey.xlsx",
                         skiprows=2)

## Joint & Marginal
pd.crosstab(index=consumer['Vehicle Driven'], 
            columns=consumer['Gender'], margins=True,
            normalize=True)
# Conditional probabilities for every row
pd.crosstab(index=consumer['Vehicle Driven'], 
            columns=consumer['Gender'], margins=True,
            normalize='index')

# Conditional probabilities for every column
pd.crosstab(index=consumer['Vehicle Driven'], 
            columns=consumer['Gender'], margins=True,
            normalize='columns')

####
a = (4.35+21.74+17.39+4.35)/100
b = (4.35+4.35+4.35+17.39)/100
c = (17.39+21.74+17.39+4.35)/100
d = 1 - (17.39+21.74+17.39+4.35+4.35+4.35+4.35)/100
