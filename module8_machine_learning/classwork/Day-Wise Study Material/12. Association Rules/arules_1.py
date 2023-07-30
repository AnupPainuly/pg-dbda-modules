import pandas as pd
import matplotlib.pylab as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

fp_df = pd.read_csv('Faceplate.csv',index_col=0)

fp_df = fp_df.astype(bool)
# create frequent itemsets
itemsets = apriori(fp_df, min_support=0.2,
                   use_colnames=True)
print(itemsets)
# and convert into rules
rules = association_rules(itemsets, metric='confidence', 
                          min_threshold=0.6)
rules = rules[['antecedents','consequents',
               'support','confidence','lift']]
rules.sort_values(by='lift', ascending = False)

############## Cosmetics ##################################

fp_df = pd.read_csv('Cosmetics.csv',index_col=0)

fp_df = fp_df.astype(bool)
# create frequent itemsets
itemsets = apriori(fp_df, min_support=0.2,
                   use_colnames=True)
print(itemsets)
# and convert into rules
rules = association_rules(itemsets, metric='confidence', 
                          min_threshold=0.6)
rules = rules[['antecedents','consequents',
               'support','confidence','lift']]
rules.sort_values(by='lift', ascending = False)

################ Groceries ##############################
from mlxtend.preprocessing import TransactionEncoder

groceries = []
with open("groceries.csv","r") as f:groceries = f.read()
groceries = groceries.split("\n")

groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))

te = TransactionEncoder()
te_ary = te.fit_transform(groceries_list)
fp_df = pd.DataFrame(te_ary, columns=te.columns_)

fp_df = fp_df.astype(bool)
# create frequent itemsets
itemsets = apriori(fp_df, min_support=0.01,
                   use_colnames=True)
print(itemsets)
# and convert into rules
rules = association_rules(itemsets, metric='confidence', 
                          min_threshold=0.5)
rules = rules[['antecedents','consequents',
               'support','confidence','lift']]
rules.sort_values(by='confidence', ascending = False).iloc[0,:]

################## DatasetA ###############################

groceries = []
with open("DatasetA.csv","r") as f:groceries = f.read()
groceries = groceries.split("\n")

groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))

te = TransactionEncoder()
te_ary = te.fit_transform(groceries_list)
fp_df = pd.DataFrame(te_ary, columns=te.columns_)

fp_df = fp_df.iloc[:,1:]

fp_df = fp_df.astype(bool)
# create frequent itemsets
itemsets = apriori(fp_df, min_support=0.01,
                   use_colnames=True)
print(itemsets)
# and convert into rules
rules = association_rules(itemsets, metric='confidence', 
                          min_threshold=0.5)
rules = rules[['antecedents','consequents',
               'support','confidence','lift']]
rules.sort_values(by='confidence', ascending = False).iloc[10,:]
